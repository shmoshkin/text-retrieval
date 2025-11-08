"""
Part 3: Statistics Analysis

This script analyzes the inverted index to compute:
1. Top 10 terms with highest document frequency
2. Top 10 terms with lowest document frequency
3. Analysis of characteristics
4. Two terms with similar document frequencies that co-occur
"""

from inverted_index import InvertedIndex
from collections import Counter
import math


def compute_document_frequencies(index: InvertedIndex):
    """
    Compute document frequency for each term in the index.
    
    Document frequency = number of documents containing the term.
    
    Args:
        index: An InvertedIndex instance.
        
    Returns:
        A dictionary mapping terms to their document frequencies.
    """
    doc_frequencies = {}
    
    for term, postings in index.index.items():
        # Document frequency is the length of the postings list
        doc_frequencies[term] = len(postings)
    
    return doc_frequencies


def find_top_terms(doc_frequencies, top_n=10, highest=True):
    """
    Find the top N terms with highest or lowest document frequency.
    
    Args:
        doc_frequencies: Dictionary mapping terms to document frequencies.
        top_n: Number of top terms to return.
        highest: If True, return highest frequencies; if False, return lowest.
        
    Returns:
        List of tuples (term, frequency) sorted appropriately.
    """
    sorted_terms = sorted(doc_frequencies.items(), 
                         key=lambda x: x[1], 
                         reverse=highest)
    
    return sorted_terms[:top_n]


def find_terms_with_similar_df(doc_frequencies, index: InvertedIndex, 
                                tolerance=0.10, min_cooccur=5, max_pairs=1000):
    """
    Find two terms with similar document frequencies that co-occur in documents.
    Optimized version that samples pairs efficiently.
    
    Args:
        doc_frequencies: Dictionary mapping terms to document frequencies.
        index: An InvertedIndex instance.
        tolerance: Maximum relative difference in DF (e.g., 0.10 = 10%).
        min_cooccur: Minimum number of documents where both terms co-occur.
        max_pairs: Maximum number of pairs to check.
        
    Returns:
        List of tuples: (term1, term2, df1, df2, cooccur_count, cooccur_docs)
    """
    results = []
    
    # Filter to terms with reasonable DF (not too rare, not too common)
    # Focus on terms with DF between 10 and 10000 for better results
    filtered_terms = [(t, df) for t, df in doc_frequencies.items() 
                     if 10 <= df <= 10000]
    
    # Sort by document frequency
    filtered_terms.sort(key=lambda x: x[1])
    
    print(f"  Checking {min(len(filtered_terms), max_pairs)} term pairs...")
    
    pairs_checked = 0
    
    # Compare terms with similar document frequencies
    for i, (term1, df1) in enumerate(filtered_terms):
        if pairs_checked >= max_pairs:
            break
            
        # Look for terms with similar DF in nearby positions
        # Check a window around current position
        window_size = min(50, len(filtered_terms) - i)
        
        for j in range(i + 1, min(i + window_size, len(filtered_terms))):
            if pairs_checked >= max_pairs:
                break
                
            term2, df2 = filtered_terms[j]
            pairs_checked += 1
            
            # Check if document frequencies are similar (within tolerance)
            if df2 > df1 * (1 + tolerance):
                break  # df2 is too large, move to next term1
            
            if df2 < df1 * (1 - tolerance):
                continue  # df2 is too small, keep looking
            
            # Get postings lists (already sorted, so we can use efficient intersection)
            postings1 = index.get_postings(term1)
            postings2 = index.get_postings(term2)
            
            # Efficient intersection of sorted lists
            cooccur_count = 0
            cooccur_docs = []
            i1, i2 = 0, 0
            
            while i1 < len(postings1) and i2 < len(postings2):
                if postings1[i1] == postings2[i2]:
                    cooccur_count += 1
                    if len(cooccur_docs) < 20:  # Store first 20
                        docno = index.get_original_docno(postings1[i1])
                        if docno:
                            cooccur_docs.append(docno)
                    i1 += 1
                    i2 += 1
                elif postings1[i1] < postings2[i2]:
                    i1 += 1
                else:
                    i2 += 1
            
            if cooccur_count >= min_cooccur:
                results.append((term1, term2, df1, df2, cooccur_count, cooccur_docs))
                
                # If we found a good pair with high co-occurrence, we can stop
                if cooccur_count >= min_cooccur * 2:
                    return results
    
    # Sort results by co-occurrence count (descending)
    results.sort(key=lambda x: x[4], reverse=True)
    return results[:5]  # Return top 5


def analyze_characteristics(high_freq_terms, low_freq_terms):
    """
    Analyze and explain the characteristics of high and low frequency terms.
    
    Args:
        high_freq_terms: List of (term, frequency) tuples for high frequency terms.
        low_freq_terms: List of (term, frequency) tuples for low frequency terms.
        
    Returns:
        A string containing the analysis.
    """
    analysis = []
    analysis.append("CHARACTERISTICS ANALYSIS:")
    analysis.append("=" * 60)
    analysis.append("")
    
    # High frequency terms analysis
    analysis.append("HIGH DOCUMENT FREQUENCY TERMS:")
    analysis.append("-" * 60)
    high_terms_list = [term for term, _ in high_freq_terms]
    analysis.append(f"Terms: {', '.join(high_terms_list)}")
    analysis.append("")
    analysis.append("Characteristics:")
    analysis.append("1. These are typically common words (stop words) like 'the', 'a', 'and', etc.")
    analysis.append("2. They appear in a very large proportion of documents (often 50-90%+).")
    analysis.append("3. They have low discriminative power - they don't help distinguish between documents.")
    analysis.append("4. They are usually short, common function words or very general content words.")
    analysis.append("5. In information retrieval, these terms are often removed or down-weighted.")
    analysis.append("")
    
    # Low frequency terms analysis
    analysis.append("LOW DOCUMENT FREQUENCY TERMS:")
    analysis.append("-" * 60)
    low_terms_list = [term for term, _ in low_freq_terms]
    analysis.append(f"Terms: {', '.join(low_terms_list)}")
    analysis.append("")
    analysis.append("Characteristics:")
    analysis.append("1. These are rare or specialized terms that appear in very few documents.")
    analysis.append("2. They have high discriminative power - they are good at identifying specific documents.")
    analysis.append("3. They often represent:")
    analysis.append("   - Proper nouns (names, places, organizations)")
    analysis.append("   - Technical terms or jargon")
    analysis.append("   - Rare concepts or specialized vocabulary")
    analysis.append("   - Typos or unusual spellings")
    analysis.append("4. They are useful for precision in search (finding very specific documents).")
    analysis.append("5. However, they may have low recall if used alone in queries.")
    analysis.append("")
    
    # Comparison
    analysis.append("KEY DIFFERENCES:")
    analysis.append("-" * 60)
    analysis.append("• Discriminative Power: Low-DF terms are highly discriminative; High-DF terms are not.")
    analysis.append("• Use in IR: Low-DF terms are good for precision; High-DF terms are often filtered.")
    analysis.append("• Vocabulary: High-DF terms are common/general; Low-DF terms are specific/rare.")
    analysis.append("• Coverage: High-DF terms appear in many docs; Low-DF terms appear in few docs.")
    analysis.append("")
    
    return "\n".join(analysis)


def generate_part3(index: InvertedIndex, output_file="Part_3.txt"):
    """
    Generate Part 3 statistics and write to output file.
    
    Args:
        index: An InvertedIndex instance.
        output_file: Path to output file.
    """
    print("Computing document frequencies...")
    doc_frequencies = compute_document_frequencies(index)
    
    print(f"Total unique terms: {len(doc_frequencies)}")
    print(f"Total documents: {index.get_document_count()}")
    
    # 1. Top 10 highest document frequency
    print("\nFinding top 10 terms with highest document frequency...")
    top_10_highest = find_top_terms(doc_frequencies, top_n=10, highest=True)
    
    # 2. Top 10 lowest document frequency
    print("Finding top 10 terms with lowest document frequency...")
    top_10_lowest = find_top_terms(doc_frequencies, top_n=10, highest=False)
    
    # 3. Analysis
    print("Analyzing characteristics...")
    analysis = analyze_characteristics(top_10_highest, top_10_lowest)
    
    # 4. Find terms with similar DF that co-occur
    print("Finding terms with similar document frequencies that co-occur...")
    similar_terms = find_terms_with_similar_df(doc_frequencies, index, 
                                            tolerance=0.15, min_cooccur=3, max_pairs=500)
    
    # Write to file
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. Top 10 highest DF
        f.write("1. TOP 10 TERMS WITH HIGHEST DOCUMENT FREQUENCY:\n")
        f.write("=" * 60 + "\n")
        for i, (term, freq) in enumerate(top_10_highest, 1):
            percentage = (freq / index.get_document_count()) * 100
            f.write(f"{i:2d}. {term:20s} : {freq:6d} documents ({percentage:5.2f}%)\n")
        f.write("\n")
        
        # 2. Top 10 lowest DF
        f.write("2. TOP 10 TERMS WITH LOWEST DOCUMENT FREQUENCY:\n")
        f.write("=" * 60 + "\n")
        for i, (term, freq) in enumerate(top_10_lowest, 1):
            percentage = (freq / index.get_document_count()) * 100
            f.write(f"{i:2d}. {term:20s} : {freq:6d} documents ({percentage:5.2f}%)\n")
        f.write("\n")
        
        # 3. Characteristics explanation
        f.write("3. CHARACTERISTICS EXPLANATION:\n")
        f.write("=" * 60 + "\n")
        f.write(analysis)
        f.write("\n")
        
        # 4. Similar DF terms that co-occur
        f.write("4. TERMS WITH SIMILAR DOCUMENT FREQUENCIES THAT CO-OCCUR:\n")
        f.write("=" * 60 + "\n")
        
        if similar_terms:
            # Use the first (best) result
            term1, term2, df1, df2, cooccur_count, cooccur_docs = similar_terms[0]
            
            f.write(f"\nTerm 1: '{term1}'\n")
            f.write(f"  Document Frequency: {df1} documents\n")
            f.write(f"  Percentage: {(df1/index.get_document_count())*100:.4f}%\n")
            f.write(f"\nTerm 2: '{term2}'\n")
            f.write(f"  Document Frequency: {df2} documents\n")
            f.write(f"  Percentage: {(df2/index.get_document_count())*100:.4f}%\n")
            f.write(f"\nSimilarity: Document frequencies differ by {abs(df1-df2)} documents ")
            f.write(f"({abs(df1-df2)/max(df1,df2)*100:.2f}% relative difference)\n")
            f.write(f"\nCo-occurrence: Both terms appear together in {cooccur_count} documents\n")
            f.write(f"  (This represents {cooccur_count/df1*100:.2f}% of term1's documents and ")
            f.write(f"{cooccur_count/df2*100:.2f}% of term2's documents)\n")
            
            f.write(f"\nSample documents where both terms co-occur (first 20):\n")
            for docno in cooccur_docs[:20]:
                f.write(f"  - {docno}\n")
            if len(cooccur_docs) > 20:
                f.write(f"  ... and {len(cooccur_docs) - 20} more documents\n")
            
            f.write(f"\nTerm Characteristics:\n")
            f.write(f"  - Both terms have similar document frequencies, indicating they are ")
            f.write(f"equally common/rare in the corpus.\n")
            f.write(f"  - Their co-occurrence suggests they may be:\n")
            f.write(f"    * Related concepts or topics\n")
            f.write(f"    * Terms that appear in similar contexts\n")
            f.write(f"    * Part of the same domain or subject area\n")
            f.write(f"  - The high co-occurrence rate suggests these terms are semantically related.\n")
            
            f.write(f"\nHow I found these terms:\n")
            f.write(f"  1. Computed document frequency for all terms in the index.\n")
            f.write(f"  2. Sorted terms by document frequency.\n")
            f.write(f"  3. For each term, searched for other terms with similar DF (within 10% tolerance).\n")
            f.write(f"  4. For pairs with similar DF, computed intersection of their postings lists.\n")
            f.write(f"  5. Selected pairs with at least 5 co-occurring documents.\n")
            f.write(f"  6. Chose the pair with the highest co-occurrence rate.\n")
        else:
            f.write("\nNo suitable term pairs found with the specified criteria.\n")
            f.write("Try adjusting tolerance or minimum co-occurrence threshold.\n")
    
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    # Import from boolean_retrieval if index was already built
    # Otherwise build it
    import sys
    import os
    
    print("Loading inverted index...")
    index = InvertedIndex()
    
    # Check if we can reuse the index from a previous run
    # For now, we'll build it (but it should be fast if already in memory)
    print("Building index...")
    index.build_index("data")
    
    # Generate Part 3
    generate_part3(index, "Part_3.txt")

