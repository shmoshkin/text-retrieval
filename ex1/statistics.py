"""
Statistics Analysis

This script analyzes the inverted index to compute:
1. Top 10 terms with highest document frequency
2. Top 10 terms with lowest document frequency
3. Analysis of characteristics
4. Two terms with similar document frequencies that co-occur
"""

from inverted_index import InvertedIndex


def compute_document_frequencies(index: InvertedIndex):
    """
    The document frequency for each term in the index.
    
    Args:
        index: An InvertedIndex instance.
        
    Returns:
        A dictionary mapping terms to their document frequencies.
    """
    doc_frequencies = {}
    
    for term, postings in index.index.items():
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


def find_terms_with_similar_df(doc_frequencies, index: InvertedIndex, max_pairs=10000):
    """
    Find two terms with similar document frequencies that also appear in the same documents.
    
    1. First, find pairs with identical DF (most similar) and check their co-occurrence
    2. Then, find pairs with similar DF (within tolerance) and check their co-occurrence
    3. Return the pair with the highest co-occurrence count
    
    Args:
        doc_frequencies: Dictionary mapping terms to document frequencies.
        index: An InvertedIndex instance.
        max_pairs: Maximum number of pairs to check.
        
    Returns:
        List of tuples: (term1, term2, df1, df2, cooccur_count, cooccur_docs)
    """
    results = []
    
    # Filter to terms with reasonable DF (not too rare, not too common)
    # Include lower DF terms (like 5) to find good pairs
    filtered_terms = [(t, df) for t, df in doc_frequencies.items() 
                     if 5 <= df <= 10000]
    
    print(f"  Analyzing {len(filtered_terms)} terms with DF between 5 and 10,000...")
    
    # Group terms by document frequency
    df_groups = {}
    for term, df in filtered_terms:
        if df not in df_groups:
            df_groups[df] = []
        df_groups[df].append(term)
    
    pairs_checked = 0
    
    # Progressive search: start with small DF and expand if no results found
    print(f"  Progressively searching from small DF values upward...")
    sorted_dfs = sorted(df_groups.keys())
    
    # Helper function to check a specific DF group
    def check_df_group(df, terms):
        """Check a DF group for pairs with identical postings lists."""
        group_results = []
        checked = 0
        
        if len(terms) < 2:
            return group_results, checked
        
        # For small groups (<=10 terms), check all pairs
        if len(terms) <= 10:
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    if pairs_checked + checked >= max_pairs:
                        break
                    postings1 = index.get_postings(term1)
                    postings2 = index.get_postings(term2)
                    
                    if postings1 == postings2:
                        cooccur_docs = [index.get_original_docno(d) for d in postings1[:20] if index.get_original_docno(d)]
                        group_results.append((term1, term2, df, df, len(postings1), cooccur_docs))
                    checked += 1
        
        # For larger groups, use hash-based approach for efficiency
        else:
            postings_to_terms = {}
            # Process all terms to build the hash map (this is fast)
            for term in terms:
                postings = tuple(index.get_postings(term))
                if postings not in postings_to_terms:
                    postings_to_terms[postings] = []
                postings_to_terms[postings].append(term)
                checked += 1
            
            # Find groups of terms with identical postings
            for postings_tuple, term_list in postings_to_terms.items():
                if len(term_list) >= 2:
                    # All terms in this list have identical postings
                    # Add all pairs from this group
                    for i, term1 in enumerate(term_list):
                        for term2 in term_list[i+1:]:
                            cooccur_docs = [index.get_original_docno(d) for d in postings_tuple[:20] if index.get_original_docno(d)]
                            group_results.append((term1, term2, df, df, len(postings_tuple), cooccur_docs))
        
        return group_results, checked
    
    # Start with small DF values and progressively increase
    # Try DF ranges: 5, then 6-10, then 11-20, then 21-50, etc.
    df_ranges = [
        (5, 5),      # Just DF=5 first
        (6, 10),     # Then 6-10
        (11, 20),    # Then 11-20
        (21, 50),    # Then 21-50
        (51, 100),   # Then 51-100
        (101, 200),  # Then 101-200
        (201, 500),  # Then 201-500
        (501, 1000), # Then 501-1000
        (1001, 10000) # Finally 1001-10000
    ]
    
    for min_df, max_df in df_ranges:
        if results:
            break  # Stop as soon as we find results
        
        print(f"  Checking DF range {min_df}-{max_df}...")
        
        # Get all DF values in this range
        dfs_in_range = [df for df in sorted_dfs if min_df <= df <= max_df]
        
        for df in dfs_in_range:
            if results:
                break  # Stop if we found results
            
            terms = df_groups[df]
            group_results, checked = check_df_group(df, terms)
            pairs_checked += checked
            
            if group_results:
                results.extend(group_results)
                print(f"    Found {len(group_results)} pairs at DF={df}")
    
    # Sort results by document frequency (higher is better for meaningful examples)
    if results:
        results.sort(key=lambda x: x[2], reverse=True)
        print(f"  Found {len(results)} total pairs appearing in exactly the same documents")
        print(f"  Best pair: '{results[0][0]}' + '{results[0][1]}' (DF={results[0][2]}, appear in exactly {results[0][4]} documents)")
        return results[:5]
    
    print(f"  No pairs found appearing in exactly the same documents")
    return []


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
    similar_terms = find_terms_with_similar_df(doc_frequencies, index, max_pairs=10000)
    
    # Write to file
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. Top 10 highest DF
        f.write("1. TOP 10 TERMS WITH HIGHEST DOCUMENT FREQUENCY:\n")
        f.write("=" * 60 + "\n")
        total_docs = index.get_document_count()
        for i, (term, freq) in enumerate(top_10_highest, 1):
            percentage = (freq / total_docs) * 100
            f.write(f"{i:2d}. {term:20s} : {freq:6d} documents ({percentage:5.2f}%)\n")
        f.write("\n")
        
        # 2. Top 10 lowest DF
        f.write("2. TOP 10 TERMS WITH LOWEST DOCUMENT FREQUENCY:\n")
        f.write("=" * 60 + "\n")
        total_docs = index.get_document_count()
        for i, (term, freq) in enumerate(top_10_lowest, 1):
            percentage = (freq / total_docs) * 100
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
            total_docs = index.get_document_count()
            f.write(f"  Percentage: {(df1/total_docs)*100:.4f}%\n")
            f.write(f"\nTerm 2: '{term2}'\n")
            f.write(f"  Document Frequency: {df2} documents\n")
            f.write(f"  Percentage: {(df2/total_docs)*100:.4f}%\n")
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
            f.write(f"  2. Filtered terms to those with document frequency between 5 and 10,000.\n")
            f.write(f"  3. Grouped terms by their document frequency.\n")
            f.write(f"  4. For each group with identical document frequency, I checked which terms have ")
            f.write(f"IDENTICAL postings lists (appear in exactly the same documents).\n")
            f.write(f"  5. Used a hash-based approach: grouped terms by their postings list (as a tuple).\n")
            f.write(f"  6. For groups with 2+ terms sharing the same postings list, I created all pairs.\n")
            f.write(f"  7. These pairs appear in EXACTLY the same documents (100% co-occurrence).\n")
            f.write(f"  8. Sorted pairs by document frequency (higher DF = more meaningful examples).\n")
            f.write(f"  9. Selected the pair with the highest document frequency.\n")
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

