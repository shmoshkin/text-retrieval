import os
import pickle
import json
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search import (
    get_topics_with_reader,
)
from pyserini.search.lucene import LuceneSearcher
from pyserini.vectorizer import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from scipy import sparse

# Load TSV-format topics
topic_file_path = "./files/queriesROBUST.txt"
index_path = f"./index/RobustPyserini"

topics = get_topics_with_reader(
    "io.anserini.search.topicreader.TsvIntTopicReader", topic_file_path
)
# fix query ids
queries = {}
for topic_id, topic in topics.items():
    fixed_topic_id = str(topic_id)
    if len(fixed_topic_id) == 2:
        fixed_topic_id = "0" + str(topic_id)
    queries[fixed_topic_id] = topic["title"]
print(len(queries))
assert len(queries) == 249, "missing queries"
# sort by topic id
queries = dict(sorted(queries.items()))


def rank_documents(run_number, method="bm25", stemmer="krovetz", top_k=1000):
    """
    Rank documents using BM25, RM3, or QLD methods.
    
    Args:
        run_number: Run identifier (1, 2, or 3)
        method: "bm25", "rm3", or "qld"
        stemmer: Stemmer type (default: "krovetz")
        top_k: Number of documents to retrieve (default: 1000)
    """
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} ranking (Run {run_number})")
    print(f"{'='*60}")
    
    # Initialize the searcher with the path to your stemmed index
    searcher = LuceneSearcher(index_path)
    # specify custom analyzer for the query processing step to match the way the index was built
    analyzer = get_lucene_analyzer(
        stemmer=stemmer, stopwords=False
    )  # Ensure no stopwords are removed from the query
    searcher.set_analyzer(analyzer)

    if method == "rm3":
        searcher.set_rm3(fb_terms=500, fb_docs=5, original_query_weight=0.3)
    elif method == "bm25":
        searcher.set_bm25(k1=0.9, b=0.4)
    elif method == "qld":
        searcher.set_qld(mu=1000)
    else:
        raise Exception("Invalid ranking method")

    # Loop through each query in the topics dictionary and retrieve documents:
    results = {}  # To store results for each query
    for topic_id, topic in tqdm(queries.items(), desc=f"Processing queries ({method})", unit="query"):
        hits = searcher.search(
            topic, k=top_k
        )  # k=1000 is the number of retrieved documents
        # Store results in TREC format for each topic
        results[topic_id] = [(hit.docid, hit.lucene_docid, i+1, hit.score) for i, hit in enumerate(hits)]

    # Now you can save the results to a file in the TREC format:
    output_file = f"./results/run_{run_number}_{method}.res"
    if not os.path.exists("./results"):
        os.makedirs("./results")
    sorted_results = dict(sorted(results.items()))
    with open(output_file, "w") as f:
        for topic_id, hits in sorted_results.items():
            for rank, (docid, _, _, score) in enumerate(hits, start=1):
                f.write(f"{topic_id} Q0 {docid} {rank} {score:.4f} run{run_number}\n")
    
    print(f"✅ Results saved to: {output_file}")
    print(f"   Total queries processed: {len(sorted_results)}")
    print(f"   Documents per query: {top_k}")
    return output_file


def rank_documents_vector(run_number, top_k=1000, stemmer="krovetz"):
    """
    Document ranking using sparse matrix operations and precomputations.
    
    Args:
        run_number: Run identifier (1, 2, or 3)
        top_k: Number of documents to retrieve (default: 1000)
        stemmer: Stemmer type (default: "krovetz")
    """
    print(f"\n{'='*60}")
    print(f"Running Vector-based ranking (Run {run_number})")
    print(f"{'='*60}")
    
    # Initialize index reader and get document IDs
    index_reader = IndexReader(index_path)
    num_docs = index_reader.stats()["documents"]
    print(f"Total documents in index: {num_docs:,}")
    
    all_docids = [
        index_reader.convert_internal_docid_to_collection_docid(i)
        for i in tqdm(range(num_docs), desc="Loading document IDs", unit="doc")
    ]

    # Initialize vectorizer and load/create document matrix
    doc_vectorizer = TfidfVectorizer(lucene_index_path=index_path, verbose=True)
    doc_matrix_file = f"./results/doc_matrix_{run_number}.npz"
    doc_norms_file = f"./results/doc_norms_{run_number}.npy"

    if not os.path.exists(doc_matrix_file):
        # Build and save document matrix and norms
        print("Building document matrix...")
        doc_matrix = doc_vectorizer.get_vectors(all_docids)
        sparse.save_npz(doc_matrix_file, doc_matrix)
        print(f"✅ Document matrix saved: {doc_matrix_file}")

        # Precompute document norms
        print("Precomputing document norms...")
        doc_norms = np.sqrt(np.array(doc_matrix.power(2).sum(axis=1)).ravel())
        doc_norms[doc_norms == 0] = 1e-10  # Prevent division by zero
        np.save(doc_norms_file, doc_norms)
        print(f"✅ Document norms saved: {doc_norms_file}")
    else:
        # Load precomputed data
        print("Loading precomputed data...")
        doc_matrix = sparse.load_npz(doc_matrix_file)
        doc_norms = np.load(doc_norms_file)
        print(f"✅ Loaded cached matrix: {doc_matrix.shape}")

    # Process queries
    results = {}
    query_list = list(queries.items())

    print("Retrieving documents for each query...")
    for topic_id, query_text in tqdm(query_list, desc="Processing queries (vector)", unit="query"):
        # Vectorize query
        query_vector = doc_vectorizer.get_query_vector(query_text)
        if query_vector.nnz == 0:  # Handle empty queries
            results[topic_id] = []
            continue

        query_sparse = query_vector

        # cosine similarities
        qnorm = sparse.linalg.norm(query_sparse)
        if qnorm < 1e-10:
            similarity_scores = np.zeros(num_docs)
        else:
            # Matrix multiplication for all documents 
            scores = query_sparse.dot(doc_matrix.T).toarray().ravel()
            similarity_scores = scores / (qnorm * doc_norms)

        # top_k 
        ranked_indices = np.argsort(-similarity_scores)[:top_k]
        top_k_scores = similarity_scores[ranked_indices]
        top_k_docids = [all_docids[i] for i in ranked_indices]

        results[topic_id] = [
            (docid, rank, float(score))
            for rank, (docid, score) in enumerate(
                zip(top_k_docids, top_k_scores), start=1
            )
        ]

    # Save
    output_file = f"./results/run_{run_number}_vector.res"
    os.makedirs("./results", exist_ok=True)

    with open(output_file, "w") as fout:
        for topic_id in sorted(results.keys()):
            for rank, (docid, _, score) in enumerate(results[topic_id], start=1):
                fout.write(
                    f"{topic_id} Q0 {docid} {rank} {score:.4f} run{run_number}\n"
                )

    print(f"\n✅ Vector-based run file saved to: {output_file}")
    print(f"   Total queries processed: {len(results)}")
    print(f"   Documents per query: {top_k}")
    return output_file


def run_all_methods(run_rm3=True, run_vector=True, run_bm25=False):
    """
    Run all ranking methods.
    
    Args:
        run_rm3: If True, run RM3 method (Run 1)
        run_vector: If True, run Vector-based method (Run 2)
        run_bm25: If True, run BM25 method (optional, for testing)
    
    Returns:
        Dictionary with method names as keys and output file paths as values
    """
    results = {}
    
    if run_rm3:
        results['rm3'] = rank_documents(run_number=1, method="rm3")
    
    if run_vector:
        results['vector'] = rank_documents_vector(run_number=2, top_k=1000)
    
    if run_bm25:
        results['bm25'] = rank_documents(run_number=10, method="bm25")
    
    return results


if __name__ == "__main__":
    # Configuration: Set which methods to run
    RUN_RM3 = True      # Run 1: RM3 method
    RUN_VECTOR = True  # Run 2: Vector-based method
    RUN_BM25 = False   # Optional: BM25 baseline (for testing)
    
    # Run selected methods
    run_all_methods(run_rm3=RUN_RM3, run_vector=RUN_VECTOR, run_bm25=RUN_BM25)
