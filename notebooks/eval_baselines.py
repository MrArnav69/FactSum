"""
Baseline Model Evaluation Script
================================
Generates summaries from flat (non-hierarchical) baselines and evaluates
with all factuality metrics for paper comparison table.

Baselines:
1. PEGASUS-Multi (google/pegasus-multi_news) - Standard flat summarizer
2. PRIMERA (allenai/PRIMERA) - Long-document summarizer with Longformer

Metrics: SummaC, FactCC, BARTScore, BERTScore, UniEval, AlignScore
"""

import torch
import numpy as np
import pandas as pd
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    LEDTokenizer,
    LEDForConditionalGeneration
)
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our metrics
import sys
sys.path.insert(0, '/Users/mrarnav69/Documents/FactSum/notebooks')
from run_publication_eval import SummaCZS, FactCC, BARTScore, BERTScoreMetric, UniEvalFact


def load_pegasus_multi():
    """Load PEGASUS-Multi_News model."""
    logger.info("Loading PEGASUS-Multi_News...")
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-multi_news")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-multi_news")
    return tokenizer, model


def load_primera():
    """Load PRIMERA model."""
    logger.info("Loading PRIMERA...")
    tokenizer = LEDTokenizer.from_pretrained("allenai/PRIMERA")
    model = LEDForConditionalGeneration.from_pretrained("allenai/PRIMERA")
    return tokenizer, model


def generate_summary(model, tokenizer, text, model_name, device='mps', max_length=256):
    """Generate summary using a flat model."""
    model = model.to(device)
    model.eval()
    
    # Truncate input to model's max length
    if 'pegasus' in model_name.lower():
        max_input = 1024
    else:  # PRIMERA can handle longer
        max_input = 4096
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input
    ).to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def evaluate_baselines(num_samples: int = 20, output_prefix: str = 'baseline_comparison'):
    """
    Generate summaries from baseline models and evaluate with factuality metrics.
    """
    print("=" * 70)
    print("BASELINE MODEL EVALUATION FOR PAPER COMPARISON")
    print("=" * 70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (same samples as hierarchical evaluation)
    print("\n[1/6] Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    samples = [dataset[int(i)] for i in indices]
    
    # Load models
    print("\n[2/6] Loading baseline models...")
    pegasus_tok, pegasus_model = load_pegasus_multi()
    primera_tok, primera_model = load_primera()
    
    # Initialize metrics
    print("\n[3/6] Initializing metrics...")
    metrics = {
        'SummaC': SummaCZS(device),
        'FactCC': FactCC(device),
        'BARTScore': BARTScore(device),
        'BERTScore': BERTScoreMetric('cpu'),
        'UniEval': UniEvalFact(device),
    }
    
    # ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Results storage
    results = {
        'PEGASUS-Multi': {'summaries': [], 'rouge': [], 'metrics': {m: [] for m in metrics}},
        'PRIMERA': {'summaries': [], 'rouge': [], 'metrics': {m: [] for m in metrics}}
    }
    
    # Generate and evaluate PEGASUS
    print("\n[4/6] Evaluating PEGASUS-Multi_News...")
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc="PEGASUS"):
        source = sample['document']
        reference = sample['summary']
        
        # Generate
        summary = generate_summary(pegasus_model, pegasus_tok, source, 'pegasus', device)
        results['PEGASUS-Multi']['summaries'].append(summary)
        
        # ROUGE
        rouge_scores = rouge.score(reference, summary)
        results['PEGASUS-Multi']['rouge'].append({
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        })
        
        # Factuality metrics
        for name, metric in metrics.items():
            try:
                score = metric.score(source, summary)
                results['PEGASUS-Multi']['metrics'][name].append(score)
            except Exception as e:
                logger.warning(f"PEGASUS {name} error: {e}")
                results['PEGASUS-Multi']['metrics'][name].append(0.0)
    
    # Generate and evaluate PRIMERA
    print("\n[5/6] Evaluating PRIMERA...")
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc="PRIMERA"):
        source = sample['document']
        reference = sample['summary']
        
        # Generate
        summary = generate_summary(primera_model, primera_tok, source, 'primera', device)
        results['PRIMERA']['summaries'].append(summary)
        
        # ROUGE
        rouge_scores = rouge.score(reference, summary)
        results['PRIMERA']['rouge'].append({
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        })
        
        # Factuality metrics
        for name, metric in metrics.items():
            try:
                score = metric.score(source, summary)
                results['PRIMERA']['metrics'][name].append(score)
            except Exception as e:
                logger.warning(f"PRIMERA {name} error: {e}")
                results['PRIMERA']['metrics'][name].append(0.0)
    
    # Aggregate results
    print("\n[6/6] Aggregating results...")
    summary_table = {}
    
    for model_name in ['PEGASUS-Multi', 'PRIMERA']:
        rouge_avg = {
            'ROUGE-1': np.mean([r['rouge1'] for r in results[model_name]['rouge']]),
            'ROUGE-2': np.mean([r['rouge2'] for r in results[model_name]['rouge']]),
            'ROUGE-L': np.mean([r['rougeL'] for r in results[model_name]['rouge']])
        }
        metric_avg = {name: np.mean(scores) for name, scores in results[model_name]['metrics'].items()}
        
        summary_table[model_name] = {**rouge_avg, **metric_avg}
    
    # Save detailed results
    for model_name in ['PEGASUS-Multi', 'PRIMERA']:
        df = pd.DataFrame({
            'generated': results[model_name]['summaries'],
            'rouge1': [r['rouge1'] for r in results[model_name]['rouge']],
            'rouge2': [r['rouge2'] for r in results[model_name]['rouge']],
            'rougeL': [r['rougeL'] for r in results[model_name]['rouge']],
            **{f'{m.lower()}_score': scores for m, scores in results[model_name]['metrics'].items()}
        })
        df.to_csv(f'{output_prefix}_{model_name.lower().replace("-", "_")}_details.csv', index=False)
    
    # Save summary
    with open(f'{output_prefix}_summary.json', 'w') as f:
        json.dump(summary_table, f, indent=2)
    
    # Print results table
    print("\n" + "=" * 90)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Model':<18} {'R-1':>8} {'R-2':>8} {'R-L':>8} {'SummaC':>8} {'FactCC':>8} {'BART':>8} {'BERT':>8} {'UniEval':>8}")
    print("-" * 90)
    
    for model_name, scores in summary_table.items():
        print(f"{model_name:<18} {scores['ROUGE-1']:>8.4f} {scores['ROUGE-2']:>8.4f} {scores['ROUGE-L']:>8.4f} "
              f"{scores['SummaC']:>8.4f} {scores['FactCC']:>8.4f} {scores['BARTScore']:>8.4f} "
              f"{scores['BERTScore']:>8.4f} {scores['UniEval']:>8.4f}")
    
    print("=" * 90)
    print(f"\nSaved: {output_prefix}_*.csv, {output_prefix}_summary.json")
    
    return summary_table


if __name__ == "__main__":
    evaluate_baselines(num_samples=20)
