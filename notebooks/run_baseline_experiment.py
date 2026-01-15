import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from hierarchical_summarizer import HierarchicalSummarizer

def run_baseline_experiment(num_samples: int = 50, device: str = 'cpu'):
    """
    Run the Baseline Experiment (Method B for the paper).
    
    Objective: 
    Measure the performance of SOTA Hierarchical Summarization 
    (Adaptive Semantic Chunking + PEGASUS) *without* Fact Verification.
    
    This establishes the 'strong baseline' that we aim to improve upon.
    """
    print(f"🚀 Starting Baseline Experiment")
    print(f"   Samples: {num_samples}")
    print(f"   Device:  {device}")
    
    # 1. Load Dataset
    print("Loading Multi-News dataset (Test Split)...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Select random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    # Convert to list of dicts for easier handling
    data_samples = [dataset[int(i)] for i in indices]
    
    # 2. Initialize Model
    print("Initializing SOTA Hierarchical Summarizer...")
    summarizer = HierarchicalSummarizer(device=device)
    
    # 3. Load Metrics
    print("Loading ROUGE metric...")
    rouge = evaluate.load('rouge')
    
    results = []
    generated_summaries = []
    reference_summaries = []
    
    # 4. Run Inference
    print("\n⚡ Generating Summaries...")
    for item in tqdm(data_samples, desc="Processing"):
        doc = item['document']
        ref = item['summary']
        
        # Run Pipeline
        try:
            output = summarizer.summarize_document(doc)
            gen_summary = output['final_summary']
            
            # Save detailed log
            results.append({
                'reference': ref,
                'generated': gen_summary,
                'num_chunks': len(output['chunks']),
                'chunk_summaries': output['chunk_summaries']
            })
            
            generated_summaries.append(gen_summary)
            reference_summaries.append(ref)
            
        except Exception as e:
            print(f"Error processing doc: {e}")
            continue
            
    # 5. Compute ROUGE
    print("\nComputing ROUGE Scores...")
    metrics = rouge.compute(
        predictions=generated_summaries, 
        references=reference_summaries,
        use_stemmer=True
    )
    
    print("\n" + "="*50)
    print("BASELINE RESULTS (Multi-News Subset)")
    print("="*50)
    print(f"ROUGE-1: {metrics['rouge1']*100:.2f}")
    print(f"ROUGE-2: {metrics['rouge2']*100:.2f}")
    print(f"ROUGE-L: {metrics['rougeL']*100:.2f}")
    print("="*50)
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv('baseline_experiment_details.csv', index=False)
    
    summary_stats = {
        'num_samples': len(results),
        'metrics': metrics,
        'config': 'Hierarchical PEGASUS (SOTA Chunking)'
    }
    with open('baseline_metrics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
        
    print(f"\nSaved detailed outputs to 'baseline_experiment_details.csv'")
    print(f"Saved metrics to 'baseline_metrics.json'")

if __name__ == "__main__":
    # Defaulting to CPU for stability based on previous testing
    # User can change to 'cuda' or 'mps' if they trust their driver stability
    run_baseline_experiment(num_samples=20, device='cpu') 
