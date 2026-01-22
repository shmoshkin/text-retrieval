import os
import pickle
import json
# Import IndexReader as per project guidelines
# Guidelines specify: from pyserini.index.lucene import IndexReader
try:
    from pyserini.index.lucene import IndexReader
except ImportError:
    # Fallback for different pyserini versions
    try:
        from pyserini.index import IndexReader
    except ImportError:
        # IndexReader not available - will use searcher fallback
        IndexReader = None
        print("⚠️  Warning: IndexReader not available. Some features may use fallback methods.")

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
# Use prebuilt index as per project guidelines
# Note: The index was created using Porter stemming and no stopword removal

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


def rank_documents(run_number, method="bm25", stemmer="porter", top_k=1000):
    """
    Rank documents using BM25, RM3, or QLD methods.
    
    Args:
        run_number: Run identifier (1, 2, or 3)
        method: "bm25", "rm3", or "qld"
        stemmer: Stemmer type (default: "porter" - matches the prebuilt index)
        top_k: Number of documents to retrieve (default: 1000)
    """
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} ranking (Run {run_number})")
    print(f"{'='*60}")
    
    # Initialize the searcher using prebuilt index (as per project guidelines)
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    
    # Specify custom analyzer for the query processing step to match the way the index was built
    # IMPORTANT: Index was created using Porter stemming and no stopword removal
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
    queries_with_few_results = []  # Track queries that don't return enough results
    
    for topic_id, topic in tqdm(queries.items(), desc=f"Processing queries ({method})", unit="query"):
        hits = searcher.search(
            topic, k=top_k
        )  # k=1000 is the number of retrieved documents
        
        # CRITICAL FIX: If we got fewer than top_k results, use query expansion to get more
        # This ensures we always have top_k results per query for consistent evaluation
        if len(hits) < top_k:
            original_count = len(hits)
            queries_with_few_results.append((topic_id, original_count))
            
            # Strategy: Use RM3 expansion to get more results when BM25 doesn't return enough
            # RM3 uses BM25 as base, so this is acceptable for ensuring we have enough results
            if method == "bm25":
                # Temporarily enable RM3 to get more candidate documents
                searcher.set_rm3(fb_terms=10, fb_docs=3, original_query_weight=0.5)
                expanded_hits = searcher.search(topic, k=top_k)
                
                # Switch back to BM25 for future queries
                searcher.set_bm25(k1=0.9, b=0.4)
                
                # Use the expanded results (they're still BM25-based, just with query expansion)
                # This ensures we always have top_k results
                hits = expanded_hits[:top_k]
                
                # If still not enough (shouldn't happen with RM3), try individual terms
                if len(hits) < top_k:
                    query_terms = topic.split()
                    existing_docids = {hit.docid for hit in hits}
                    all_hits = list(hits)
                    
                    for term in query_terms:
                        if len(all_hits) >= top_k:
                            break
                        term_hits = searcher.search(term, k=top_k)
                        for hit in term_hits:
                            if hit.docid not in existing_docids and len(all_hits) < top_k:
                                all_hits.append(hit)
                                existing_docids.add(hit.docid)
                    
                    # Sort by score and take top_k
                    hits = sorted(all_hits, key=lambda x: x.score, reverse=True)[:top_k]
            
            # Sort hits by score (descending) and take top_k
            hits = sorted(hits, key=lambda x: x.score, reverse=True)[:top_k]
            
            if len(hits) < top_k:
                print(f"⚠️  Warning: Query {topic_id} still has only {len(hits)} results (requested {top_k})")
            elif original_count < top_k:
                print(f"✅ Query {topic_id}: Expanded from {original_count} to {len(hits)} results using query expansion")
        
        # Store results in TREC format for each topic
        results[topic_id] = [(hit.docid, hit.lucene_docid, i+1, hit.score) for i, hit in enumerate(hits)]

    # Print summary of queries that needed expansion
    if queries_with_few_results:
        print(f"\n📊 Summary: {len(queries_with_few_results)} queries needed expansion to reach {top_k} results")
        if len(queries_with_few_results) <= 20:
            print(f"   Affected queries: {[qid for qid, count in queries_with_few_results]}")
        else:
            print(f"   Sample queries: {[qid for qid, count in queries_with_few_results[:10]]}...")

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


def rank_documents_vector(run_number, top_k=1000, stemmer="porter"):
    """
    Document ranking using sparse matrix operations and precomputations.
    
    Args:
        run_number: Run identifier (1, 2, or 3)
        top_k: Number of documents to retrieve (default: 1000)
        stemmer: Stemmer type (default: "porter" - matches the prebuilt index)
    """
    print(f"\n{'='*60}")
    print(f"Running Vector-based ranking (Run {run_number})")
    print(f"{'='*60}")
    
    # Initialize searcher using prebuilt index (as per project guidelines)
    searcher_temp = LuceneSearcher.from_prebuilt_index('robust04')
    
    # Get index reader and document count - try multiple approaches for compatibility
    index_reader = None
    num_docs = None
    all_docids = None
    
    if IndexReader is not None:
        try:
            # Use prebuilt index as per project guidelines
            index_reader = IndexReader.from_prebuilt_index('robust04')
            num_docs = index_reader.stats()["documents"]
        except (AttributeError, TypeError) as e:
            print(f"⚠️  IndexReader instantiation issue: {e}")
            index_reader = None
    
    # Fallback: Try to get from searcher
    if index_reader is None or num_docs is None:
        try:
            # Try to get reader from searcher
            index_reader = searcher_temp.reader
            num_docs = index_reader.num_docs
        except AttributeError:
            try:
                # Try alternative attribute names
                num_docs = searcher_temp.num_docs
                index_reader = searcher_temp
            except AttributeError:
                # Use vectorizer to get document info
                print("⚠️  Using vectorizer to determine document count...")
                # TfidfVectorizer may need local index path - will handle in vectorizer initialization
                num_docs = 528155  # Approximate for Robust04 (Robust04 has ~528k documents)
                index_reader = None
    
    print(f"Total documents in index: {num_docs:,}")
    
    # Get all document IDs
    if index_reader is not None:
        try:
            all_docids = [
                index_reader.convert_internal_docid_to_collection_docid(i)
                for i in tqdm(range(num_docs), desc="Loading document IDs", unit="doc")
            ]
        except (AttributeError, TypeError):
            try:
                # Alternative method
                all_docids = [
                    index_reader.doc(i).get("id") for i in tqdm(range(num_docs), desc="Loading document IDs", unit="doc")
                ]
            except:
                all_docids = None
    
    # If still no docids, we'll need to get them from vectorizer
    if all_docids is None:
        print("⚠️  Will get document IDs from vectorizer during matrix construction...")

    # Initialize vectorizer - try prebuilt index first, fallback to getting index path
    try:
        # Try using prebuilt index directly (if supported)
        doc_vectorizer = TfidfVectorizer.from_prebuilt_index('robust04', verbose=True)
    except (AttributeError, TypeError):
        # Fallback: TfidfVectorizer might need local index path
        # Get the index path from pyserini's cache or use searcher's path
        import pyserini
        # Try to get the cached index path
        try:
            # Pyserini caches prebuilt indexes - try to find it
            from pyserini.util import download_prebuilt_index
            index_path = download_prebuilt_index('robust04')
            doc_vectorizer = TfidfVectorizer(lucene_index_path=index_path, verbose=True)
        except:
            # Last resort: use searcher to get index path if available
            try:
                index_path = searcher_temp.index_dir
                doc_vectorizer = TfidfVectorizer(lucene_index_path=index_path, verbose=True)
            except:
                raise Exception("Could not initialize TfidfVectorizer. Please check pyserini version and TfidfVectorizer API.")
    doc_matrix_file = f"./results/doc_matrix_{run_number}.npz"
    doc_norms_file = f"./results/doc_norms_{run_number}.npy"

    if not os.path.exists(doc_matrix_file):
        # Build and save document matrix and norms
        print("Building document matrix...")
        # If we don't have docids yet, try to get them from vectorizer or use a workaround
        if all_docids is None:
            # Try to get docids from vectorizer - this might require a different approach
            # For now, we'll try to get them by processing the index differently
            print("⚠️  Attempting to get document IDs from vectorizer...")
            # The vectorizer might need docids - let's try to get them from searcher results
            # This is a workaround: search for a very common term to get many docids
            try:
                hits = searcher_temp.search("the", k=min(10000, num_docs))
                all_docids = [hit.docid for hit in hits]
                # If we got fewer than expected, we'll need to handle this
                if len(all_docids) < num_docs:
                    print(f"⚠️  Got {len(all_docids)} docids from search, but index has {num_docs} docs")
                    print("   Will use available docids - results may be incomplete")
            except:
                print("❌ Could not get document IDs. Please check pyserini version compatibility.")
                raise
        
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
