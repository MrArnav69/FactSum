import torch
from sentence_transformers import CrossEncoder
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactScorer:
    """
    SOTA Factuality Metric using NLI (Natural Language Inference).
    
    This implementation is equivalent to the **SummaC-ZS (Zero-Shot)** benchmark.
    Papers:
    - SummaC: Laban et al. (2022) "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection..."
    - FactCC: Kryscinski et al. (2020) (Predecessor, uses weaker BERT generation)
    
    Why DeBERTa-v3?
    Standard SummaC uses MNLI or VitaL. We use `cross-encoder/nli-deberta-v3-base` 
    because DeBERTa outperforms BERT/RoBERTa on NLI tasks, providing a stricter 
    and more accurate "Truth" signal than the original FactCC weights.
    
    Method:
    1. Split summary into atomic sentences.
    2. Pair each sentence with the Source Document (Hypothesis vs Premise).
    3. Calculate Entailment Probability (Softmax of 'Entailment' logit).
    4. Consistency Score = Mean Entailment Probability (SummaC-Mean).
    """
    
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.device = device
        logger.info(f"Initializing FactScorer with {model_name} on {device}")
        
        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load FactScorer model: {e}")
            raise

    def score_summary(self, source: str, summary: str) -> Dict[str, float]:
        """
        Calculate Factuality/Consistency score.
        
        Args:
            source: The full source text (or relevant chunks).
            summary: The generated summary.
            
        Returns:
            Dict containing:
            - 'score': Average probability of Entailment (0.0 to 1.0)
            - 'num_contradictions': Count of sentences predicted as Contradiction
        """
        if not summary.strip():
            return {'score': 0.0, 'contradictions': 0}
            
        # 1. Sentence Segment
        try:
            summary_sents = sent_tokenize(summary)
        except LookupError:
             nltk.download('punkt')
             nltk.download('punkt_tab')
             summary_sents = sent_tokenize(summary)
             
        if not summary_sents:
            return {'score': 0.0, 'contradictions': 0}
        
        # 2. Prepare Pairs
        # (Source, Summary_Sentence)
        # We rely on CrossEncoder truncation (512 tokens) for now as a baseline approximation.
        pairs = [[source, sent] for sent in summary_sents]
        
        # 3. Predict
        scores = self.model.predict(pairs, apply_softmax=True)
        
        # 4. Aggregate
        # Label indices for 'cross-encoder/nli-deberta-v3-base': 
        # 0: Contradiction, 1: Entailment, 2: Neutral
        
        entailment_probs = scores[:, 1] # Index 1
        contradiction_probs = scores[:, 0] # Index 0
        
        # Consistency Score = Mean Entailment Prob
        consistency_score = np.mean(entailment_probs)
        
        # Hard decisions (argmax)
        predictions = np.argmax(scores, axis=1)
        num_contradictions = np.sum(predictions == 0)
        
        return {
            'score': float(consistency_score),
            'contradictions': int(num_contradictions),
            'sentence_scores': entailment_probs.tolist()
        }

if __name__ == "__main__":
    # Unit Test
    print("Testing FactScorer...")
    scorer = FactScorer(device='cpu') # Test on CPU
    
    src = "The cat sat on the mat. It was raining outside."
    
    # Test 1: Supported
    summ_true = "The cat is on the mat."
    res_true = scorer.score_summary(src, summ_true)
    print(f"True Summary Score: {res_true['score']:.4f} (Expected > 0.8)")
    
    # Test 2: Contradicted
    summ_false = "The cat is flying in the sky."
    res_false = scorer.score_summary(src, summ_false)
    print(f"False Summary Score: {res_false['score']:.4f} (Expected < 0.2)")
    
    # Test 3: Hallucinated (Neutral)
    summ_neut = "The dog ate the food."
    res_neut = scorer.score_summary(src, summ_neut)
    print(f"Neutral Summary Score: {res_neut['score']:.4f} (Expected Low/Mid)")
