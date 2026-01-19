import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import json
import nltk

REPO_ROOT = "/Users/mrarnav69/Models and Repos"
RESULT_ROOT = "/Users/mrarnav69/Documents/FactSum/results"
os.makedirs(RESULT_ROOT, exist_ok=True)

# --- 1. DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Running on Apple Metal (MPS) Acceleration")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Running on CPU (slower).")

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('punkt_tab')

# --- 2. DATA LOADING ---
print("⏳ Loading Multi-News Dataset...")
try:
    # Loading test split
    dataset = load_dataset("Awesome075/multi_news_parquet", split='test')
    test_data = dataset.select(range(100)) 
    src_docs = test_data['document']
    gold_sums = test_data['summary']
    print(f"✅ Loaded {len(src_docs)} samples.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    sys.exit(1)

# --- 3. GENERATION FUNCTION ---
def generate_summaries(model_name, docs, device, batch_size=1):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    generated_summaries = []
    
    # Using batch_size=1 is safer for M3 memory with large contexts
    for i in tqdm(range(0, len(docs), batch_size), desc=f"Generating {model_name}"):
        batch_docs = docs[i : i + batch_size]
        max_input = 4096 if 'PRIMERA' in model_name else 1024
        
        inputs = tokenizer(batch_docs, return_tensors="pt", max_length=max_input, truncation=True, padding=True).to(device)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=256,
                length_penalty=2.0,
                early_stopping=True
            )
        
        decoded = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        generated_summaries.extend(decoded)

    # Aggressive memory cleanup for Mac
    del model
    del tokenizer
    if device.type == 'mps':
        torch.mps.empty_cache()
    return generated_summaries

# Run Generation (Check if already saved to save time)
output_file = os.path.join(RESULT_ROOT, "model_outputs.json")
if os.path.exists(output_file):
    print("📂 Found existing summaries. Loading from disk...")
    with open(output_file, 'r') as f:
        data = json.load(f)
        pegasus_preds = data['pegasus']
        primera_preds = data['primera']
else:
    pegasus_preds = generate_summaries('google/pegasus-multi_news', src_docs, device)
    primera_preds = generate_summaries('allenai/PRIMERA', src_docs, device)
    # Save immediately
    with open(output_file, 'w') as f:
        json.dump({'pegasus': pegasus_preds, 'primera': primera_preds}, f)

# --- 4. EVALUATION METRICS ---
results_data = {"Metric": [], "PEGASUS": [], "PRIMERA": []}

def add_result(metric_name, score_peg, score_prim):
    results_data["Metric"].append(metric_name)
    results_data["PEGASUS"].append(score_peg)
    results_data["PRIMERA"].append(score_prim)
    print(f"📊 {metric_name}: PEG={score_peg:.4f}, PRIM={score_prim:.4f}")

# A. ROUGE & BERTScore
print("\n--- Running ROUGE & BERTScore ---")
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

def eval_hf(preds, refs):
    r = rouge.compute(predictions=preds, references=refs)
    # NOTE: BERTScore is often faster/stable on CPU for Mac due to specific float32 requirements
    bs = bertscore.compute(predictions=preds, references=refs, lang="en", model_type="roberta-large", device="cpu") 
    return r, np.mean(bs['f1'])

peg_r, peg_bs = eval_hf(pegasus_preds, gold_sums)
prim_r, prim_bs = eval_hf(primera_preds, gold_sums)

add_result("ROUGE-1", peg_r['rouge1'], prim_r['rouge1'])
add_result("ROUGE-2", peg_r['rouge2'], prim_r['rouge2'])
add_result("ROUGE-L", peg_r['rougeL'], prim_r['rougeL'])
add_result("BERTScore", peg_bs, prim_bs)

# B. BARTScore
print("\n--- Running BARTScore ---")
sys.path.append(os.path.join(REPO_ROOT, "BARTScore"))
from bart_score import BARTScorer

# BARTScore works well on MPS
bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
def eval_bart(preds, srcs):
    return np.mean(bart_scorer.score(srcs, preds, batch_size=4))

add_result("BARTScore", eval_bart(pegasus_preds, src_docs), eval_bart(primera_preds, src_docs))

# Cleanup BART to free memory and remove from path
del bart_scorer
if 'bart_score' in sys.modules: del sys.modules['bart_score']
sys.path.remove(os.path.join(REPO_ROOT, "BARTScore"))
if device.type == 'mps': torch.mps.empty_cache()

# C. SummaC
print("\n--- Running SummaC ---")
from summac.model_summac import SummaCZS

# SummaC works on MPS
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device=device)
def eval_summac(preds, srcs):
    return np.mean(model_zs.score(srcs, preds)['scores'])

add_result("SummaC", eval_summac(pegasus_preds, src_docs), eval_summac(primera_preds, src_docs))
del model_zs
if device.type == 'mps': torch.mps.empty_cache()

# D. UniEval
print("\n--- Running UniEval ---")
unieval_path = os.path.join(REPO_ROOT, "UniEval")
sys.path.insert(0, unieval_path) # Insert at 0 to prioritize UniEval utils

# FORCE CLEANUP of old utils to prevent conflicts
if 'utils' in sys.modules: del sys.modules['utils']
if 'metric.evaluator' in sys.modules: del sys.modules['metric.evaluator']

from utils import convert_to_json
from metric.evaluator import get_evaluator

def eval_unieval(preds, srcs, refs):
    data = convert_to_json(output_list=preds, src_list=srcs, ref_list=refs)
    # UniEval generally safer on CPU for Mac if using M3, but try device first
    evaluator = get_evaluator('summarization', device=device) 
    scores = evaluator.evaluate(data, print_result=False)
    return {
        'coh': np.mean([s['coherence'] for s in scores]),
        'con': np.mean([s['consistency'] for s in scores]),
        'flu': np.mean([s['fluency'] for s in scores]),
        'rel': np.mean([s['relevance'] for s in scores])
    }

peg_uni = eval_unieval(pegasus_preds, src_docs, gold_sums)
prim_uni = eval_unieval(primera_preds, src_docs, gold_sums)

add_result("UniEval-Coherence", peg_uni['coh'], prim_uni['coh'])
add_result("UniEval-Consistency", peg_uni['con'], prim_uni['con'])
add_result("UniEval-Relevance", peg_uni['rel'], prim_uni['rel'])

# Cleanup UniEval
sys.path.remove(unieval_path)
if 'utils' in sys.modules: del sys.modules['utils']
if device.type == 'mps': torch.mps.empty_cache()

# E. AlignScore
print("\n--- Running AlignScore ---")
# AlignScore installed via pip in setup_repos.py, so we import normally
from alignscore import AlignScore

def eval_align(preds, srcs):
    ckpt = os.path.join(REPO_ROOT, "AlignScore/AlignScore-base.ckpt")
    # Use CPU for AlignScore on Mac to prevent RoBERTa precision issues
    scorer = AlignScore(model='roberta-base', batch_size=16, device='cpu', ckpt_path=ckpt, evaluation_mode='nli_sp')
    return np.mean(scorer.score(contexts=srcs, claims=preds))

add_result("AlignScore", eval_align(pegasus_preds, src_docs), eval_align(primera_preds, src_docs))

# --- 5. SAVE RESULTS ---
df_final = pd.DataFrame(results_data)
csv_path = os.path.join(RESULT_ROOT, "final_results_mac.csv")
df_final.to_csv(csv_path, index=False)

print("\n" + "="*50)
print("✅ EVALUATION COMPLETE")
print(f"📄 Results saved to: {csv_path}")
print("="*50)
print(df_final)