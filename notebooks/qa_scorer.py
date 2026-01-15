import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import logging
import nltk
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAScorer:
    """
    QA-Based Factuality Metric (Implementing QAGS/QAFactEval logic).
    
    Logic:
    1. Question Generation (QG): Generate candidate questions from the SUMMARY.
    2. Question Answering (QA): Answer these questions using the SOURCE.
    3. Verification: Answer these questions using the SUMMARY (to verify relevance).
    4. Scoring: Check if the Source-Answer matches the Summary-Answer (using F1/Exact Match).
    
    If Source answers the question effectively the same way the Summary does, it's Factual.
    If Source usually returns "No Answer" or a different answer, it's a Hallucination.
    
    Note: This is computationally expensive (2 models involved).
    """
    
    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        logger.info(f"Initializing QAScorer on {device}...")
        
        # 1. Load Question Generation Model (T5)
        # We use a popular lightweight QG model
        self.qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap" 
        logger.info(f"Loading QG model: {self.qg_model_name}")
        self.qg_tokenizer = AutoTokenizer.from_pretrained(self.qg_model_name)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(self.qg_model_name).to(self.device)
        
        # 2. Load Question Answering Model (RoBERTa SQuAD)
        self.qa_model_name = "deepset/roberta-base-squad2"
        logger.info(f"Loading QA model: {self.qa_model_name}")
        self.qa_pipeline = pipeline('question-answering', model=self.qa_model_name, tokenizer=self.qa_model_name, device=0 if device=='cuda' else (-1))
        # Note: 'pipeline' on MPS logic might differ, usually -1 forces CPU which is safe but slow. 
        # If MPS is available, we try to use it manually or let pipeline handle it.
        # HF Pipeline device handling: device=0 (cuda:0), device=-1 (cpu). device='mps' not always supported in pipeline args directly in older versions.
        # We'll stick to CPU for QA pipeline to avoid MPS bugs unless user has specific transformers version, 
        # OR we can pass the model object directly if we loaded it to MPS.
        
        if device == 'mps':
            # Manual load for MPS support in pipeline
            qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name).to('mps')
            qa_tok = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tok, device='mps')

    def generate_questions(self, text: str) -> List[str]:
        """Generate questions from summary text."""
        # Simple heuristic: T5 QG usually needs "answer context". 
        # For full QAGS we extract Nouns/Entities as answers and ask format "answer: X context: Y".
        # Simplification: We will extract Noun Phrases (NPs) as candidates.
        
        try:
            sentences = sent_tokenize(text)
        except:
             nltk.download('punkt')
             nltk.download('punkt_tab')
             sentences = sent_tokenize(text)
             
        questions = []
        
        for sent in sentences:
            inputs = self.qg_tokenizer.encode("generate questions: " + sent, return_tensors="pt").to(self.device)
            # Generate
            outputs = self.qg_model.generate(inputs, max_length=64, num_beams=4, num_return_sequences=1)
            q = self.qg_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Filter bad questions
            if q and "?" in q:
                questions.append(q)
                
        return questions

    def compute_f1(self, a_gold, a_pred):
        """Compute F1 score between two strings."""
        def normalize_answer(s):
            """Lower text and remove punctuation, articles and extra whitespace."""
            import string, re
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))

        gold_toks = normalize_answer(a_gold).split()
        pred_toks = normalize_answer(a_pred).split()
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def score_summary(self, source: str, summary: str) -> Dict[str, float]:
        """
        1. Generate questions from Summary.
        2. Answer them using Source. (A_source)
        3. Answer them using Summary. (A_summary)
        4. Matches = F1(A_source, A_summary)
        """
        import collections # inner import for utility
        
        # 1. Generate Questions
        # We need "answers" to generate questions typically. 
        # For this lightweight version, we let the model hallucinate questions from the sentence context directly
        # (Using the "generate questions: " prompt mode of mrm8488 model)
        questions = self.generate_questions(summary)
        
        if not questions:
            return {'score': 0.0, 'num_questions': 0}
            
        f1_scores = []
        
        for q in questions:
            # Answer from Source
            try:
                # Truncate source for QA model limit (usually 512). 
                # Ideally we scan chunks. For baseline, we use first 1000 chars roughly.
                # Or usage stride logic. Pipeline handles stride if configured, but slow.
                ans_source = self.qa_pipeline(question=q, context=source[:2000], handle_impossible_answer=True)
                a_src_text = ans_source['answer']
                
                # Answer from Summary (Control)
                ans_summary = self.qa_pipeline(question=q, context=summary, handle_impossible_answer=True)
                a_sum_text = ans_summary['answer']
                
                # Compare
                # We reuse the compute_f1 logic inside
                # (Quick inline implementation to avoid dependency issues)
                
                # If Q is bad and both return empty, that's a match, but trivially.
                # If Summary has answer but Source doesn't -> Hallucination.
                
                # Normalize
                def normalize(s):
                    import string, re
                    s = s.lower()
                    s = "".join(ch for ch in s if ch not in string.punctuation)
                    return " ".join(s.split())
                
                norm_src = normalize(a_src_text)
                norm_sum = normalize(a_sum_text)
                
                if not norm_sum: 
                    # Summary couldn't answer its own question? Bad question. Skip.
                    continue
                    
                if norm_src == norm_sum:
                    f1_scores.append(1.0)
                elif norm_sum in norm_src: # Loose match
                    f1_scores.append(1.0)
                else:
                    # Overlap check
                    toks_src = set(norm_src.split())
                    toks_sum = set(norm_sum.split())
                    if not toks_sum:
                        match = 0.0
                    else:
                        overlap = toks_src.intersection(toks_sum)
                        match = len(overlap) / len(toks_sum) # Recall style
                    f1_scores.append(match)
                    
            except Exception as e:
                logger.warning(f"QA Error: {e}")
                continue
                
        if not f1_scores:
            return {'score': 0.0, 'num_questions': 0}
            
        final_score = float(np.mean(f1_scores))
        
        return {
            'score': final_score,
            'num_questions': len(questions),
            'questions': questions
        }

if __name__ == "__main__":
    # Test
    scorer = QAScorer(device='cpu')
    src = "Albert Einstein was born in Germany in 1879. He developed the theory of relativity."
    summ = "Einstein was born in Germany."
    
    print("Testing QA Scorer...")
    res = scorer.score_summary(src, summ)
    print(f"Score: {res['score']} (Questions: {res['num_questions']})")
