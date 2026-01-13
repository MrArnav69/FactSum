import os
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Import your chunker
from semantic_document_chunker import SemanticDocumentChunker

# Import HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError('Please install datasets: pip install datasets')


def main():
    print('\n===== Loading multi_news_parquet dataset =====\n')
    multi_news = load_dataset("Awesome075/multi_news_parquet", split="test")
    test_doc = multi_news[0]['document']

    print('\n===== Testing SemanticDocumentChunker =====\n')

    # Initialize chunker
    chunker = SemanticDocumentChunker(max_tokens=300, overlap_tokens=50)

    # Test single document chunking
    print('-> Chunking single document...')
    chunks = chunker.chunk_document(test_doc)
    print(f'Number of chunks created: {len(chunks)}')
    print('First chunk preview:', chunks[0]['text'][:200], '...\n')

    # Test semantic coherence computation
    print('-> Computing semantic coherence of first chunk...')
    first_chunk_sentences = chunks[0]['sentences']
    coherence = chunker.compute_semantic_coherence(first_chunk_sentences, global_coherence=True)
    print('Coherence metrics:', coherence, '\n')

    # Test validation
    print('-> Validating chunks...')
    valid, warnings_list = chunker.validate_chunks(chunks)
    print('Is valid:', valid)
    print('Warnings:', warnings_list[:5] if warnings_list else 'None', '\n')

    # Test summary statistics
    print('-> Generating summary statistics...')
    stats = chunker.get_summary_statistics(chunks)
    for k, v in stats.items():
        print(f'{k}: {v}')
    print()

    # Test batch chunking with first 3 documents
    print('-> Testing batch chunking with first 3 documents...')
    docs = [multi_news[i]['document'] for i in range(3)]
    all_chunks = []
    for doc in docs:
        all_chunks.append(chunker.chunk_document(doc))
    print(f'Batch chunking completed: {len(all_chunks)} documents processed\n')

    # Test ablation modes comparison
    print('-> Comparing ablation modes...')
    ablation_results = chunker.compare_ablation_modes([test_doc],
                                                      modes=[None, 'no_semantic', 'no_overlap', 'fixed_overlap'])
    for mode, res in ablation_results.items():
        print(f'Mode: {mode}, Avg num chunks: {res["num_chunks_avg"]}, '
              f'Avg local coherence: {res.get("avg_local_coherence", 0):.3f}')
    print()

    # Test cache clearing and performance stats
    print('-> Performance stats before clearing cache...')
    perf_stats = chunker.get_performance_stats()
    print(perf_stats, '\n')

    print('-> Clearing cache...')
    chunker.clear_cache()
    print('Cache cleared.\n')

    # Optional: visualize heatmap (saved to file)
    try:
        print('-> Generating coherence heatmap...')
        output_path = os.path.join(os.getcwd(), 'coherence_heatmap.png')
        fig = chunker.visualize_coherence_heatmap(chunks, output_path=output_path)
        print(f'Heatmap saved to: {output_path}')
    except Exception as e:
        print('Heatmap generation failed:', e)


if __name__ == '__main__':
    main()