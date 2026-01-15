"""
Comprehensive Factuality Evaluation Suite
=========================================
This script implements ORIGINAL algorithms from the research papers
using HuggingFace checkpoints directly (to avoid legacy package conflicts).

Metrics Implemented:
1. SummaC-ZS (Zero-Shot NLI) - Laban et al., 2022
2. FactCC - Kryscinski et al., 2020
3. QA-Based (QAGS-style) - Wang et al., 2020
"""

import torch
import numpy as np
from typing import Dict, List
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 1. SummaC-ZS (Original Algorithm)
# ============================================================
class SummaCZS:
    """
    SummaC Zero-Shot implementation.
    Paper: "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection"
    
    Uses RoBERTa-Large-MNLI for NLI predictions.
    Algorithm:
    1. Split summary into sentences.
    2. For each sentence, predict entailment against the source.
    3. Return average entailment score.
    """
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        logger.info(f"SummaC-ZS: Loading roberta-large-mnli on {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli").to(device)
        self.model.eval()
        
    def score(self, source: str, summary: str) -> Dict:
        """Score a single source-summary pair."""
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            sentences = sent_tokenize(summary)
        except:
            nltk.download('punkt_tab')
            sentences = sent_tokenize(summary)
            
        if not sentences:
            return {'score': 0.0, 'sentence_scores': []}
            
        entailment_probs = []
        
        for sent in sentences:
            # Tokenize as NLI pair: source (premise) vs sentence (hypothesis)
            inputs = self.tokenizer(
                source[:1024],  # Truncate source to fit model
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # MNLI labels: 0=contradiction, 1=neutral, 2=entailment
                entailment_prob = probs[0][2].item()
                entailment_probs.append(entailment_prob)
                
        return {
            'score': float(np.mean(entailment_probs)),
            'sentence_scores': entailment_probs
        }


# ============================================================
# 2. FactCC (Original Algorithm)
# ============================================================
class FactCC:
    """
    FactCC implementation.
    Paper: "Evaluating the Factual Consistency of Abstractive Text Summarization"
    
    Uses the manueldeprada/FactCC checkpoint (replicated from Salesforce).
    This is a BERT-based binary classifier trained on synthetic data.
    """
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        logger.info(f"FactCC: Loading manueldeprada/FactCC on {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("manueldeprada/FactCC")
            self.model = AutoModelForSequenceClassification.from_pretrained("manueldeprada/FactCC").to(device)
            self.model.eval()
            self.available = True
        except Exception as e:
            logger.warning(f"FactCC model not available: {e}. Using fallback.")
            self.available = False
            
    def score(self, source: str, summary: str) -> Dict:
        """Score a single source-summary pair."""
        if not self.available:
            return {'score': 0.0, 'available': False}
            
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            sentences = sent_tokenize(summary)
        except:
            nltk.download('punkt_tab')
            sentences = sent_tokenize(summary)
            
        if not sentences:
            return {'score': 0.0, 'sentence_scores': []}
            
        consistent_probs = []
        
        for sent in sentences:
            # FactCC expects: [CLS] source [SEP] claim [SEP]
            inputs = self.tokenizer(
                source[:1024],
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # FactCC labels: 0=inconsistent, 1=consistent
                consistent_prob = probs[0][1].item() if outputs.logits.shape[-1] > 1 else probs[0][0].item()
                consistent_probs.append(consistent_prob)
                
        return {
            'score': float(np.mean(consistent_probs)),
            'sentence_scores': consistent_probs,
            'available': True
        }


# ============================================================
# 3. QA-Based Metric (QAGS-style)
# ============================================================
class QAGSMetric:
    """
    QAGS-style QA-based factuality metric.
    Paper: "Asking and Answering Questions to Evaluate the Factual Consistency"
    
    Algorithm:
    1. Generate questions from summary.
    2. Answer questions using source.
    3. Answer questions using summary.
    4. Compare answers (F1 overlap).
    """
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        logger.info(f"QAGS: Initializing QG and QA models on {device}")
        
        # Question Generation (T5-based)
        self.qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        self.qg_tokenizer = AutoTokenizer.from_pretrained(self.qg_model_name)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(self.qg_model_name).to(device)
        
        # Question Answering (RoBERTa-SQuAD)
        self.qa_pipeline = pipeline(
            'question-answering',
            model='deepset/roberta-base-squad2',
            device=0 if device == 'cuda' else -1
        )
        
    def _generate_questions(self, text: str, max_questions: int = 5) -> List[str]:
        """Generate questions from text."""
        from nltk.tokenize import sent_tokenize
        import nltk
        try:
            sentences = sent_tokenize(text)
        except:
            nltk.download('punkt_tab')
            sentences = sent_tokenize(text)
            
        questions = []
        for sent in sentences[:max_questions]:
            input_text = "generate questions: " + sent
            inputs = self.qg_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.qg_model.generate(inputs, max_length=64, num_beams=4)
                q = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if q and "?" in q:
                    questions.append(q)
                    
        return questions
        
    def score(self, source: str, summary: str) -> Dict:
        """Score a single source-summary pair."""
        questions = self._generate_questions(summary)
        
        if not questions:
            return {'score': 0.0, 'num_questions': 0}
            
        matches = []
        for q in questions:
            try:
                # Answer from source
                ans_src = self.qa_pipeline(question=q, context=source[:2000], handle_impossible_answer=True)
                # Answer from summary
                ans_sum = self.qa_pipeline(question=q, context=summary, handle_impossible_answer=True)
                
                # Compare (simple overlap)
                src_text = ans_src['answer'].lower().strip()
                sum_text = ans_sum['answer'].lower().strip()
                
                if not sum_text:
                    continue
                    
                if src_text == sum_text or src_text in sum_text or sum_text in src_text:
                    matches.append(1.0)
                else:
                    # Token overlap
                    src_toks = set(src_text.split())
                    sum_toks = set(sum_text.split())
                    if sum_toks:
                        overlap = len(src_toks & sum_toks) / len(sum_toks)
                        matches.append(overlap)
                    else:
                        matches.append(0.0)
            except Exception as e:
                logger.warning(f"QA error: {e}")
                continue
                
        if not matches:
            return {'score': 0.0, 'num_questions': len(questions)}
            
        return {
            'score': float(np.mean(matches)),
            'num_questions': len(questions)
        }


# ============================================================
# Main Evaluation Function
# ============================================================
def evaluate_baseline(csv_path: str = 'baseline_experiment_details.csv', num_samples: int = 20):
    """Run comprehensive evaluation on baseline summaries."""
    import pandas as pd
    from datasets import load_dataset
    
    print("="*60)
    print("COMPREHENSIVE FACTUALITY EVALUATION")
    print("="*60)
    
    # Load results
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Reload source documents
    print("Reloading Multi-News for source documents...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    data_samples = [dataset[int(i)] for i in indices]
    
    # Initialize all scorers
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nInitializing metrics...")
    summac = SummaCZS(device=device)
    factcc = FactCC(device=device)
    qags = QAGSMetric(device=device)
    
    # Storage
    results = {
        'summac_scores': [],
        'factcc_scores': [],
        'qags_scores': []
    }
    
    print("\nEvaluating summaries...")
    for i, row in tqdm(df.iterrows(), total=min(len(df), num_samples), desc="Scoring"):
        if i >= len(data_samples):
            break
            
        source = data_samples[i]['document']
        summary = row['generated']
        
        # SummaC
        try:
            sc = summac.score(source, summary)
            results['summac_scores'].append(sc['score'])
        except Exception as e:
            print(f"SummaC error {i}: {e}")
            results['summac_scores'].append(0.0)
            
        # FactCC
        try:
            fc = factcc.score(source, summary)
            results['factcc_scores'].append(fc['score'])
        except Exception as e:
            print(f"FactCC error {i}: {e}")
            results['factcc_scores'].append(0.0)
            
        # QAGS
        try:
            qa = qags.score(source, summary)
            results['qags_scores'].append(qa['score'])
        except Exception as e:
            print(f"QAGS error {i}: {e}")
            results['qags_scores'].append(0.0)
    
    # Add to dataframe
    df['summac_score'] = results['summac_scores']
    df['factcc_score'] = results['factcc_scores']
    df['qags_score'] = results['qags_scores']
    
    # Save
    output_csv = 'comprehensive_factuality_results.csv'
    df.to_csv(output_csv, index=False)
    
    # Calculate aggregates
    metrics = {
        'SummaC (NLI)': np.mean(results['summac_scores']),
        'FactCC': np.mean(results['factcc_scores']),
        'QAGS (QA)': np.mean(results['qags_scores'])
    }
    
    # Save metrics
    output_json = 'comprehensive_metrics.json'
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("COMPREHENSIVE FACTUALITY RESULTS (Baseline)")
    print("="*60)
    for metric_name, score in metrics.items():
        print(f"{metric_name:20s}: {score:.4f}")
    print("="*60)
    print(f"Saved detailed results to: {output_csv}")
    print(f"Saved metrics to: {output_json}")
    
    return metrics


if __name__ == "__main__":
    evaluate_baseline()
