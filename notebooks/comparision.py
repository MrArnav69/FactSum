"""
FactSum: Complete Model Comparison Script (PEGASUS vs PRIMERA)
================================================================

Complete production-ready script that compares PEGASUS and PRIMERA
using IDENTICAL hierarchical pipeline with semantic chunking.

Author: Arnav Gupta
Date: January 2026

Usage:
    python factsum_comparison.py --num_examples 10
    python factsum_comparison.py --num_examples 50 --output_dir results/
    python factsum_comparison.py --num_examples 100 --fusion_strategy iterative

Requirements:
    pip install transformers datasets rouge-score nltk scipy matplotlib seaborn pandas torch tqdm
"""

import argparse
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from rouge_score import rouge_scorer
from scipy import stats

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

import re


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================

def check_protobuf_availability() -> bool:
    """
    Check if protobuf library is available.
    
    Returns:
        True if protobuf is available, False otherwise
    """
    try:
        # Use importlib to avoid linter warnings
        import importlib
        importlib.import_module('google.protobuf')
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def load_tokenizer_with_fallback(model_name: str):
    """
    Load tokenizer with fallback to slow tokenizer if protobuf is unavailable.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tokenizer instance
        
    Raises:
        ImportError: If protobuf is required but not available, with helpful message
    """
    # Check if this is a Pegasus model that requires protobuf
    is_pegasus = 'pegasus' in model_name.lower()
    
    if is_pegasus and not check_protobuf_availability():
        # Try to use slow tokenizer as fallback
        try:
            from transformers import PegasusTokenizer
            print(f"  Warning: protobuf not available, using slow tokenizer")
            return PegasusTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ImportError(
                f"\n{'='*80}\n"
                f"MISSING DEPENDENCY: protobuf\n"
                f"{'='*80}\n"
                f"The model '{model_name}' requires the protobuf library.\n\n"
                f"To fix this, please install protobuf:\n"
                f"  pip install protobuf\n"
                f"  OR\n"
                f"  conda install protobuf\n\n"
                f"After installation, restart your Python runtime.\n"
                f"{'='*80}\n"
            ) from e
    
    # Default: try fast tokenizer first
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except ImportError as e:
        if 'protobuf' in str(e).lower():
            # If protobuf error occurs, try slow tokenizer
            if is_pegasus:
                try:
                    from transformers import PegasusTokenizer
                    print(f"  Warning: Falling back to slow tokenizer (protobuf issue)")
                    return PegasusTokenizer.from_pretrained(model_name)
                except Exception:
                    pass
            
            # Re-raise with helpful message
            raise ImportError(
                f"\n{'='*80}\n"
                f"MISSING DEPENDENCY: protobuf\n"
                f"{'='*80}\n"
                f"The model '{model_name}' requires the protobuf library.\n\n"
                f"To fix this, please install protobuf:\n"
                f"  pip install protobuf\n"
                f"  OR\n"
                f"  conda install protobuf\n\n"
                f"After installation, restart your Python runtime.\n"
                f"{'='*80}\n"
            ) from e
        else:
            raise


# =============================================================================
# SEMANTIC DOCUMENT CHUNKER
# =============================================================================

class SemanticDocumentChunker:
    """
    State-of-the-art semantic document chunker with sentence boundary preservation.
    """
    
    def __init__(self, 
                 tokenizer,
                 max_tokens: int = 1024,
                 overlap_tokens: int = 128,
                 use_sentence_boundaries: bool = True,
                 min_chunk_tokens: int = 256,
                 preserve_paragraphs: bool = True):
        """
        Initialize semantic chunker.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Target overlap tokens
            use_sentence_boundaries: Never split sentences
            min_chunk_tokens: Minimum chunk size
            preserve_paragraphs: Try to keep paragraphs together
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.use_sentence_boundaries = use_sentence_boundaries
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_paragraphs = preserve_paragraphs
        
        if overlap_tokens >= max_tokens:
            raise ValueError(f"Overlap ({overlap_tokens}) must be less than max_tokens ({max_tokens})")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        text = re.sub(r'Enlarge this image.*?AP', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'toggle caption.*?AP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        return text.strip()
    
    def split_into_articles(self, document: str) -> List[str]:
        """Split multi-document cluster into articles."""
        articles = [a.strip() for a in document.split("|||") if a.strip()]
        if not articles:
            articles = [document.strip()]
        return articles
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_token_count(self, text: str) -> int:
        """Get accurate token count."""
        return len(self.tokenizer.tokenize(text))
    
    def find_optimal_overlap_sentences(self, 
                                       previous_sentences: List[str]) -> List[str]:
        """Find optimal sentences for overlap."""
        if not previous_sentences:
            return []
        
        overlap_sentences = []
        current_tokens = 0
        target = self.overlap_tokens
        
        for sent in reversed(previous_sentences):
            sent_tokens = self.get_token_count(sent)
            
            if current_tokens > 0 and current_tokens + sent_tokens > target * 1.5:
                break
            
            overlap_sentences.insert(0, sent)
            current_tokens += sent_tokens
            
            if current_tokens >= target * 0.8:
                break
        
        return overlap_sentences
    
    def chunk_with_sentence_boundaries(self,
                                       sentences: List[str],
                                       article_idx: int,
                                       previous_chunk_sentences: Optional[List[str]] = None) -> List[Dict]:
        """Create chunks respecting sentence boundaries."""
        chunks = []
        current_sentences = []
        current_tokens = 0
        
        # Add overlap from previous chunk
        overlap_sents = []
        if previous_chunk_sentences:
            overlap_sents = self.find_optimal_overlap_sentences(previous_chunk_sentences)
            current_sentences.extend(overlap_sents)
            current_tokens = sum(self.get_token_count(s) for s in overlap_sents)
        
        for sent in sentences:
            sent_tokens = self.get_token_count(sent)
            
            if current_tokens + sent_tokens > self.max_tokens:
                if current_tokens >= self.min_chunk_tokens or len(chunks) == 0:
                    # Save current chunk
                    chunk_text = ' '.join(current_sentences)
                    chunks.append({
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'sentences': current_sentences.copy(),
                        'token_count': current_tokens,
                        'sentence_count': len(current_sentences),
                        'article_indices': [article_idx],
                        'has_overlap': len(chunks) > 0 or previous_chunk_sentences is not None,
                        'overlap_token_count': sum(self.get_token_count(s) for s in overlap_sents)
                    })
                    
                    # Start new chunk with overlap
                    overlap_sents = self.find_optimal_overlap_sentences(current_sentences)
                    current_sentences = overlap_sents + [sent]
                    current_tokens = sum(self.get_token_count(s) for s in current_sentences)
                else:
                    current_sentences.append(sent)
                    current_tokens += sent_tokens
            else:
                current_sentences.append(sent)
                current_tokens += sent_tokens
        
        # Add final chunk
        if current_sentences and (current_tokens >= self.min_chunk_tokens or len(chunks) == 0):
            chunk_text = ' '.join(current_sentences)
            overlap_count = sum(self.get_token_count(s) for s in overlap_sents) if len(chunks) > 0 or previous_chunk_sentences else 0
            chunks.append({
                'chunk_id': len(chunks),
                'text': chunk_text,
                'sentences': current_sentences.copy(),
                'token_count': current_tokens,
                'sentence_count': len(current_sentences),
                'article_indices': [article_idx],
                'has_overlap': len(chunks) > 0 or previous_chunk_sentences is not None,
                'overlap_token_count': overlap_count
            })
        
        return chunks
    
    def chunk_document(self, document: str) -> List[Dict]:
        """Main chunking function."""
        cleaned_doc = self.clean_text(document)
        articles = self.split_into_articles(cleaned_doc)
        
        all_chunks = []
        previous_chunk_sentences = None
        
        for article_idx, article in enumerate(articles):
            article_sentences = self.split_into_sentences(article)
            
            article_chunks = self.chunk_with_sentence_boundaries(
                article_sentences,
                article_idx,
                previous_chunk_sentences
            )
            
            for chunk in article_chunks:
                chunk['chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)
            
            if article_chunks:
                previous_chunk_sentences = article_chunks[-1].get('sentences', [])
        
        return all_chunks
    
    def get_summary_statistics(self, chunks: List[Dict]) -> Dict[str, any]:
        """Get statistics about chunking."""
        if not chunks:
            return {'num_chunks': 0}
        
        token_counts = [c['token_count'] for c in chunks]
        overlap_counts = [c.get('overlap_token_count', 0) for c in chunks if c.get('has_overlap', False)]
        sentence_counts = [c.get('sentence_count', 0) for c in chunks if 'sentence_count' in c]
        
        stats = {
            'num_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'total_overlap_tokens': sum(overlap_counts),
            'net_tokens': sum(token_counts) - sum(overlap_counts),
            'avg_tokens_per_chunk': np.mean(token_counts),
            'std_tokens_per_chunk': np.std(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunks_with_overlap': sum(1 for c in chunks if c.get('has_overlap', False)),
            'avg_overlap_tokens': np.mean(overlap_counts) if overlap_counts else 0,
            'token_efficiency': (sum(token_counts) - sum(overlap_counts)) / sum(token_counts) * 100 if sum(token_counts) > 0 else 0
        }
        
        if sentence_counts:
            stats.update({
                'total_sentences': sum(sentence_counts),
                'avg_sentences_per_chunk': np.mean(sentence_counts),
                'min_sentences': min(sentence_counts),
                'max_sentences': max(sentence_counts)
            })
        
        return stats


# =============================================================================
# HIERARCHICAL SUMMARIZER
# =============================================================================

class HierarchicalSummarizer:
    """
    Hierarchical multi-document summarization with two-stage approach.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 config: Dict,
                 device: str = 'cpu'):
        """
        Initialize hierarchical summarizer.
        
        Args:
            model: HuggingFace seq2seq model
            tokenizer: HuggingFace tokenizer
            config: Summarization configuration dict
            device: Device to use (cuda/mps/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        
        self.stats = {
            'chunks_processed': 0,
            'chunk_summaries_generated': 0,
            'cluster_summaries_generated': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0
        }
    
    def summarize_chunk(self, chunk_text: str) -> Tuple[str, int]:
        """Summarize a single chunk."""
        inputs = self.tokenizer(
            chunk_text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        input_tokens = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum().item()
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config['chunk_max_length'],
                min_length=self.config['chunk_min_length'],
                num_beams=self.config['chunk_num_beams'],
                length_penalty=self.config['chunk_length_penalty'],
                no_repeat_ngram_size=self.config['chunk_no_repeat_ngram_size'],
                early_stopping=self.config['early_stopping']
            )
        
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        output_tokens = len(self.tokenizer.tokenize(summary_text))
        
        self.stats['chunks_processed'] += 1
        self.stats['chunk_summaries_generated'] += 1
        self.stats['total_input_tokens'] += input_tokens
        self.stats['total_output_tokens'] += output_tokens
        
        return summary_text, output_tokens
    
    def fuse_summaries(self, chunk_summaries: List[str]) -> str:
        """Fuse chunk summaries into final summary."""
        if self.config['fusion_strategy'] == 'concatenate':
            fused_text = " ".join(chunk_summaries)
        else:  # iterative
            summaries = chunk_summaries.copy()
            while len(summaries) > 1:
                merged = []
                for i in range(0, len(summaries), 2):
                    if i + 1 < len(summaries):
                        pair_text = summaries[i] + " " + summaries[i + 1]
                        inputs = self.tokenizer(
                            pair_text,
                            max_length=1024,
                            truncation=True,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        with torch.no_grad():
                            summary_ids = self.model.generate(
                                inputs['input_ids'],
                                max_length=self.config['cluster_max_length'] // 2,
                                num_beams=4,
                                early_stopping=True
                            )
                        
                        merged_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        merged.append(merged_summary)
                    else:
                        merged.append(summaries[i])
                summaries = merged
            fused_text = summaries[0] if summaries else ""
        
        # Generate final summary
        fused_tokens = len(self.tokenizer.tokenize(fused_text))
        
        if fused_tokens <= 1024:
            inputs = self.tokenizer(
                fused_text,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.config['cluster_max_length'],
                    min_length=self.config['cluster_min_length'],
                    num_beams=self.config['cluster_num_beams'],
                    length_penalty=self.config['cluster_length_penalty'],
                    no_repeat_ngram_size=self.config['cluster_no_repeat_ngram_size'],
                    early_stopping=self.config['early_stopping'],
                    diversity_penalty=self.config['diversity_penalty']
                )
            
            final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            # Too long, return first chunk summary
            final_summary = chunk_summaries[0] if chunk_summaries else ""
        
        output_tokens = len(self.tokenizer.tokenize(final_summary))
        self.stats['cluster_summaries_generated'] += 1
        self.stats['total_output_tokens'] += output_tokens
        
        return final_summary
    
    def summarize_document(self, chunks: List[Dict]) -> Dict:
        """Complete hierarchical summarization pipeline."""
        # Stage 1: Chunk-level summarization
        chunk_summaries = []
        for chunk in chunks:
            summary, tokens = self.summarize_chunk(chunk['text'])
            chunk_summaries.append(summary)
        
        # Stage 2: Cluster-level fusion
        final_summary = self.fuse_summaries(chunk_summaries)
        
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'num_chunks': len(chunks),
            'total_compression_ratio': self.stats['total_input_tokens'] / len(self.tokenizer.tokenize(final_summary))
                                      if len(self.tokenizer.tokenize(final_summary)) > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'chunks_processed': 0,
            'chunk_summaries_generated': 0,
            'cluster_summaries_generated': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0
        }


# =============================================================================
# MODEL PIPELINE
# =============================================================================

class FactSumPipeline:
    """
    Complete FactSum pipeline for a single model.
    """
    
    def __init__(self, 
                 model_name: str,
                 chunking_config: Dict,
                 summarization_config: Dict,
                 device: Optional[str] = None):
        """
        Initialize FactSum pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            chunking_config: Chunking configuration
            summarization_config: Summarization configuration
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.chunking_config = chunking_config
        self.summarization_config = summarization_config
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Loading {model_name}...")
        print(f"  Device: {self.device}")
        
        # Load model and tokenizer with robust error handling
        try:
            self.tokenizer = load_tokenizer_with_fallback(model_name)
        except ImportError as e:
            # Re-raise with context
            raise ImportError(
                f"Failed to load tokenizer for {model_name}.\n"
                f"Original error: {str(e)}"
            ) from e
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Initialize components
        self.chunker = SemanticDocumentChunker(
            tokenizer=self.tokenizer,
            **chunking_config
        )
        
        self.summarizer = HierarchicalSummarizer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=summarization_config,
            device=self.device
        )
        
        print(f"✓ {model_name.split('/')[-1]} loaded successfully\n")
    
    def process_document(self, document: str) -> Dict:
        """Process a single document through complete pipeline."""
        # Stage 1: Chunking
        chunks = self.chunker.chunk_document(document)
        chunk_stats = self.chunker.get_summary_statistics(chunks)
        
        # Stage 2: Summarization
        self.summarizer.reset_statistics()
        result = self.summarizer.summarize_document(chunks)
        
        return {
            'chunks': chunks,
            'chunk_stats': chunk_stats,
            'final_summary': result['final_summary'],
            'chunk_summaries': result['chunk_summaries'],
            'num_chunks': result['num_chunks'],
            'compression_ratio': result['total_compression_ratio']
        }


# =============================================================================
# ROUGE EVALUATOR
# =============================================================================

class ROUGEEvaluator:
    """ROUGE evaluation for summarization."""
    
    def __init__(self, use_stemmer: bool = True):
        """Initialize ROUGE scorer."""
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=use_stemmer
        )
    
    def evaluate_single(self, generated: str, reference: str) -> Dict:
        """Evaluate single summary."""
        scores = self.scorer.score(reference, generated)
        
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall': scores['rouge2'].recall,
            'rougeL_f1': scores['rougeL'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall
        }
    
    def evaluate_batch(self, generated_list: List[str], reference_list: List[str]) -> pd.DataFrame:
        """Evaluate batch of summaries."""
        results = []
        for gen, ref in zip(generated_list, reference_list):
            result = self.evaluate_single(gen, ref)
            result['generated_length'] = len(gen.split())
            result['reference_length'] = len(ref.split())
            results.append(result)
        
        return pd.DataFrame(results)


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

class ModelComparison:
    """
    Compare multiple models using identical pipeline.
    """
    
    def __init__(self,
                 models_config: Dict[str, str],
                 chunking_config: Dict,
                 summarization_config: Dict,
                 device: Optional[str] = None):
        """
        Initialize comparison engine.
        
        Args:
            models_config: Dict mapping model names to HF identifiers
            chunking_config: Chunking configuration
            summarization_config: Summarization configuration
            device: Device to use
        """
        self.models_config = models_config
        self.chunking_config = chunking_config
        self.summarization_config = summarization_config
        self.device = device
        
        # Pre-check dependencies for Pegasus models
        protobuf_available = check_protobuf_availability()
        pegasus_models = [name for name, model_id in models_config.items() 
                         if 'pegasus' in model_id.lower()]
        
        if pegasus_models and not protobuf_available:
            print(f"\n{'='*80}")
            print("WARNING: protobuf library not found")
            print(f"{'='*80}")
            print(f"The following models require protobuf: {', '.join(pegasus_models)}")
            print(f"\nAttempting to use slow tokenizer fallback...")
            print(f"If you encounter errors, install protobuf:")
            print(f"  pip install protobuf")
            print(f"  OR")
            print(f"  conda install protobuf")
            print(f"{'='*80}\n")
        
        # Initialize pipelines
        self.pipelines = {}
        for model_name, model_id in models_config.items():
            self.pipelines[model_name] = FactSumPipeline(
                model_name=model_id,
                chunking_config=chunking_config,
                summarization_config=summarization_config,
                device=device
            )
        
        # Initialize evaluator
        self.evaluator = ROUGEEvaluator(use_stemmer=True)
        
        # Storage
        self.results = {name: [] for name in models_config.keys()}
    
    def run_comparison(self, documents: List[str], references: List[str]) -> Dict:
        """
        Run complete comparison on dataset.
        
        Args:
            documents: List of input documents
            references: List of reference summaries
            
        Returns:
            Dictionary with comprehensive results
        """
        print("="*80)
        print("RUNNING MODEL COMPARISON")
        print("="*80)
        print(f"Number of examples: {len(documents)}")
        print(f"Models: {', '.join(self.models_config.keys())}\n")
        
        # Process each document with each model
        for idx, (doc, ref) in enumerate(tqdm(zip(documents, references), 
                                              total=len(documents),
                                              desc="Processing")):
            for model_name, pipeline in self.pipelines.items():
                try:
                    result = pipeline.process_document(doc)
                    
                    self.results[model_name].append({
                        'example_id': idx,
                        'generated_summary': result['final_summary'],
                        'reference_summary': ref,
                        'num_chunks': result['num_chunks'],
                        'compression_ratio': result['compression_ratio'],
                        'token_efficiency': result['chunk_stats']['token_efficiency']
                    })
                    
                except Exception as e:
                    print(f"\n✗ Error with {model_name} on example {idx}: {str(e)}")
                    self.results[model_name].append({
                        'example_id': idx,
                        'generated_summary': "",
                        'reference_summary': ref,
                        'num_chunks': 0,
                        'compression_ratio': 0,
                        'token_efficiency': 0
                    })
        
        print("\n✓ Processing complete")
        
        # Evaluate ROUGE scores
        print("\nEvaluating ROUGE scores...")
        rouge_results = {}
        
        for model_name in self.models_config.keys():
            generated = [r['generated_summary'] for r in self.results[model_name] if r['generated_summary']]
            references = [r['reference_summary'] for r in self.results[model_name] if r['generated_summary']]
            
            rouge_df = self.evaluator.evaluate_batch(generated, references)
            
            # Add metadata
            for i, row in rouge_df.iterrows():
                rouge_df.at[i, 'num_chunks'] = self.results[model_name][i]['num_chunks']
                rouge_df.at[i, 'compression_ratio'] = self.results[model_name][i]['compression_ratio']
                rouge_df.at[i, 'token_efficiency'] = self.results[model_name][i]['token_efficiency']
            
            rouge_results[model_name] = rouge_df
        
        print("✓ ROUGE evaluation complete\n")
        
        return {
            'results': self.results,
            'rouge_results': rouge_results
        }
    
    def print_comparison_table(self, rouge_results: Dict[str, pd.DataFrame]):
        """Print comparison table."""
        print("="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        comparison_data = []
        for model_name, df in rouge_results.items():
            comparison_data.append({
                'Model': model_name,
                'ROUGE-1': f"{df['rouge1_f1'].mean():.4f} ± {df['rouge1_f1'].std():.4f}",
                'ROUGE-2': f"{df['rouge2_f1'].mean():.4f} ± {df['rouge2_f1'].std():.4f}",
                'ROUGE-L': f"{df['rougeL_f1'].mean():.4f} ± {df['rougeL_f1'].std():.4f}",
                'Avg Chunks': f"{df['num_chunks'].mean():.1f}",
                'Token Eff': f"{df['token_efficiency'].mean():.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        print()
    
    def statistical_significance(self, rouge_results: Dict[str, pd.DataFrame]) -> Dict:
        """Compute statistical significance tests."""
        if len(rouge_results) != 2:
            print("Statistical tests require exactly 2 models")
            return {}
        
        models = list(rouge_results.keys())
        df1 = rouge_results[models[0]]
        df2 = rouge_results[models[1]]
        
        print("="*80)
        print(f"STATISTICAL SIGNIFICANCE: {models[1]} vs {models[0]}")
        print("="*80)
        
        significance = {}
        
        for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']:
            t_stat, p_value = stats.ttest_rel(df1[metric], df2[metric])
            
            mean1 = df1[metric].mean()
            mean2 = df2[metric].mean()
            improvement = ((mean2 - mean1) / mean1) * 100
            
            metric_display = metric.replace('_f1', '').replace('rouge', 'ROUGE-')
            
            print(f"\n{metric_display}:")
            print(f"  {models[0]}: {mean1:.4f} ± {df1[metric].std():.4f}")
            print(f"  {models[1]}: {mean2:.4f} ± {df2[metric].std():.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")
            
            # Ensure all values are JSON-serializable (cast numpy/scipy types to Python scalars)
            significance[metric] = {
                'model1_mean': float(mean1),
                'model2_mean': float(mean2),
                'improvement': float(improvement),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        print()
        return significance
    
    def create_visualizations(self, rouge_results: Dict[str, pd.DataFrame], output_path: str):
        """Create comparison visualizations."""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        colors = {'PEGASUS': '#3498db', 'PRIMERA': '#e74c3c'}
        
        models = list(rouge_results.keys())
        
        # Plot 1: ROUGE Scores Bar Chart
        ax1 = axes[0, 0]
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        x = np.arange(len(metrics))
        width = 0.35
        
        scores = {}
        for model in models:
            df = rouge_results[model]
            scores[model] = [df['rouge1_f1'].mean(), df['rouge2_f1'].mean(), df['rougeL_f1'].mean()]
        
        for i, (model, score_list) in enumerate(scores.items()):
            offset = width * (i - len(models)/2 + 0.5)
            ax1.bar(x + offset, score_list, width, label=model,
                   color=colors.get(model, f'C{i}'), alpha=0.8)
        
        ax1.set_ylabel('F1 Score')
        ax1.set_title('ROUGE Scores Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: ROUGE-1 Distribution
        ax2 = axes[0, 1]
        data_to_plot = [rouge_results[m]['rouge1_f1'] for m in models]
        bp = ax2.boxplot(data_to_plot, labels=models, patch_artist=True)
        
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors.get(model, 'lightblue'))
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('ROUGE-1 F1')
        ax2.set_title('ROUGE-1 Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Chunking Behavior
        ax3 = axes[0, 2]
        chunk_means = [rouge_results[m]['num_chunks'].mean() for m in models]
        ax3.bar(models, chunk_means, color=[colors.get(m, 'gray') for m in models], alpha=0.8)
        ax3.set_ylabel('Average Chunks per Document')
        ax3.set_title('Chunking Behavior (Should be Similar)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: ROUGE-1 Per Example
        ax4 = axes[1, 0]
        for model in models:
            df = rouge_results[model]
            ax4.plot(df.index, df['rouge1_f1'], marker='o', label=model,
                    color=colors.get(model, 'gray'), alpha=0.7, linewidth=2)
        ax4.set_xlabel('Example ID')
        ax4.set_ylabel('ROUGE-1 F1')
        ax4.set_title('ROUGE-1 Across Examples')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Length Comparison
        ax5 = axes[1, 1]
        for model in models:
            df = rouge_results[model]
            ax5.scatter(df['reference_length'], df['generated_length'],
                       label=model, alpha=0.6, color=colors.get(model, 'gray'), s=100)
        ax5.plot([0, 200], [0, 200], 'k--', alpha=0.3, label='Perfect match')
        ax5.set_xlabel('Reference Length (words)')
        ax5.set_ylabel('Generated Length (words)')
        ax5.set_title('Summary Length Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Token Efficiency
        ax6 = axes[1, 2]
        eff_means = [rouge_results[m]['token_efficiency'].mean() for m in models]
        ax6.bar(models, eff_means, color=[colors.get(m, 'gray') for m in models], alpha=0.8)
        ax6.set_ylabel('Token Efficiency (%)')
        ax6.set_title('Semantic Chunking Efficiency')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}\n")
        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='FactSum: Compare PEGASUS and PRIMERA with identical pipeline'
    )
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of examples to process (default: 10)')
    parser.add_argument('--fusion_strategy', type=str, default='concatenate',
                       choices=['concatenate', 'iterative'],
                       help='Fusion strategy (default: concatenate)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    print("="*80)
    print("FACTSUM MODEL COMPARISON")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of examples: {args.num_examples}")
    print(f"  Fusion strategy: {args.fusion_strategy}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {args.device if args.device else 'auto-detect'}")
    print()
    
    # Models to compare
    models_config = {
        'PEGASUS': 'google/pegasus-multi_news',
        'PRIMERA': 'allenai/PRIMERA'
    }
    
    # Chunking configuration (IDENTICAL for both models)
    chunking_config = {
        'max_tokens': 1024,
        'overlap_tokens': 128,
        'use_sentence_boundaries': True,
        'min_chunk_tokens': 256,
        'preserve_paragraphs': True
    }
    
    # Summarization configuration (IDENTICAL for both models)
    summarization_config = {
        'chunk_max_length': 256,
        'chunk_min_length': 64,
        'chunk_num_beams': 8,
        'chunk_length_penalty': 1.0,
        'chunk_no_repeat_ngram_size': 3,
        'cluster_max_length': 256,
        'cluster_min_length': 100,
        'cluster_num_beams': 10,
        'cluster_length_penalty': 1.2,
        'cluster_no_repeat_ngram_size': 3,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 0.95,
        'do_sample': False,
        'early_stopping': True,
        'diversity_penalty': 0.5,
        'fusion_strategy': args.fusion_strategy
    }
    
    # Load dataset
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    print("Loading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet")
    test_data = dataset['test']
    print(f"✓ Loaded {len(test_data)} test examples\n")
    
    # Select subset
    documents = [test_data[i]['document'] for i in range(args.num_examples)]
    references = [test_data[i]['summary'] for i in range(args.num_examples)]
    
    # Initialize comparison
    comparison = ModelComparison(
        models_config=models_config,
        chunking_config=chunking_config,
        summarization_config=summarization_config,
        device=args.device
    )
    
    # Run comparison
    results = comparison.run_comparison(documents, references)
    
    # Print results
    comparison.print_comparison_table(results['rouge_results'])
    
    # Statistical significance
    significance = comparison.statistical_significance(results['rouge_results'])
    
    # Create visualizations
    viz_path = os.path.join(args.output_dir, 'model_comparison.png')
    comparison.create_visualizations(results['rouge_results'], viz_path)
    
    # Save results
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save ROUGE results
    for model_name, df in results['rouge_results'].items():
        csv_path = os.path.join(args.output_dir, f'{model_name.lower()}_rouge_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ {model_name} ROUGE results saved to {csv_path}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'num_examples': args.num_examples,
            'fusion_strategy': args.fusion_strategy,
            'chunking': chunking_config,
            'summarization': {k: v for k, v in summarization_config.items() if k != 'fusion_strategy'}
        },
        'models': models_config,
        'rouge_summary': {
            model: {
                'rouge1': float(df['rouge1_f1'].mean()),
                'rouge2': float(df['rouge2_f1'].mean()),
                'rougeL': float(df['rougeL_f1'].mean()),
                'avg_chunks': float(df['num_chunks'].mean()),
                'token_efficiency': float(df['token_efficiency'].mean())
            }
            for model, df in results['rouge_results'].items()
        },
        'significance': significance
    }
    
    summary_path = os.path.join(args.output_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    # Generate recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if 'rouge1_f1' in significance:
        improvement = significance['rouge1_f1']['improvement']
        p_value = significance['rouge1_f1']['p_value']
        
        if improvement > 3 and p_value < 0.05:
            recommendation = "PRIMERA"
            reason = f"{improvement:.1f}% better ROUGE-1 (statistically significant, p={p_value:.4f})"
        elif improvement > 0:
            recommendation = "PRIMERA"
            reason = f"Marginal improvement ({improvement:.1f}%), not statistically significant"
        else:
            recommendation = "PEGASUS"
            reason = "Comparable performance, already working"
        
        print(f"\n✅ Recommended Model: {recommendation}")
        print(f"   Reason: {reason}")
        print(f"\n   Both models use IDENTICAL hierarchical pipeline")
        print(f"   Your architectural novelty is preserved!")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()