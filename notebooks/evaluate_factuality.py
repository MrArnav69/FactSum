import pandas as pd
import numpy as np
from datasets import load_dataset
from fact_scorer import FactScorer
from tqdm import tqdm
import torch

def evaluate_factuality(csv_path: str = 'baseline_experiment_details.csv', num_samples: int = 20):
    """
    Evaluate Factuality of existing baseline results.
    Recovers source docs by reloading dataset with seed 42.
    """
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Reloading Multi-News dataset to recover source documents...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Re-select same indices
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    data_samples = [dataset[int(i)] for i in indices]
    
    print("Initializing Scorers...")
    # Use CUDA if available, else MPS, else CPU
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Init Models
    fact_scorer = FactScorer(device=device)
    try:
        from qa_scorer import QAScorer
        qa_scorer = QAScorer(device=device)
        use_qa = True
    except Exception as e:
        print(f"Warning: Could not load QAScorer ({e}). Skipping QA metrics.")
        use_qa = False
    
    nli_scores = []
    contradictions = []
    qa_scores = []
    
    print("Scoring Summaries (NLI + QA)...")
    
    if len(df) != len(data_samples):
        print(f"WARNING: CSV has {len(df)} rows but we reloaded {len(data_samples)} samples.")
        print("Truncating to defined num_samples...")
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Fact Checking"):
        if i >= len(data_samples):
            break
            
        generated_summary = row['generated']
        source_doc = data_samples[i]['document']
        
        # 1. NLI Score (SummaC)
        try:
            nli_res = fact_scorer.score_summary(source_doc, generated_summary)
            nli_scores.append(nli_res['score'])
            contradictions.append(nli_res['contradictions'])
        except Exception as e:
            print(f"NLI Error row {i}: {e}")
            nli_scores.append(0.0)
            contradictions.append(0)
            
        # 2. QA Score (QAGS/QAFactEval)
        if use_qa:
            try:
                qa_res = qa_scorer.score_summary(source_doc, generated_summary)
                qa_scores.append(qa_res['score'])
            except Exception as e:
                print(f"QA Error row {i}: {e}")
                qa_scores.append(0.0)
        else:
            qa_scores.append(0.0)

    # Add columns
    df['nli_score'] = nli_scores
    df['num_contradictions'] = contradictions
    if use_qa:
        df['qa_score'] = qa_scores
    
    # Save
    output_path = 'baseline_factuality_results.csv'
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("MULTI-METRIC RESULTS (Baseline)")
    print("="*50)
    print(f"Avg NLI Score (SummaC): {np.mean(nli_scores):.4f}")
    if use_qa:
        print(f"Avg QA Score (QAGS):    {np.mean(qa_scores):.4f}")
    print(f"Total Contradictions:   {np.sum(contradictions)}")
    print("="*50)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    evaluate_factuality()
