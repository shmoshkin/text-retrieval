"""
Boolean Retrieval Model Implementation

This module implements a BooleanRetrieval class that processes Boolean queries
in Reverse Polish Notation (RPN) and retrieves matching documents using an
inverted index.

The implementation uses efficient algorithms that leverage the fact that
postings lists are sorted by internal document IDs, avoiding the use of
set data structures for set operations.
"""

from typing import List, Set
from inverted_index import InvertedIndex


class BooleanRetrieval:
    """
    A class to process Boolean queries and retrieve matching documents.
    
    Boolean queries are expressed in Reverse Polish Notation (RPN) with
    operators: AND, OR, and NOT (where NOT is treated as AND NOT).
    
    The implementation uses efficient merge-based algorithms for set operations
    on sorted postings lists, as required by the assignment.
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        """
        Initialize BooleanRetrieval with an inverted index.
        
        Args:
            inverted_index: An InvertedIndex instance containing the indexed corpus.
        """
        self.index = inverted_index
        self.total_docs = inverted_index.get_document_count()
    
    def _intersect(self, list1: List[int], list2: List[int]) -> List[int]:
        """
        Compute the intersection of two sorted postings lists.
        
        Uses a two-pointer merge algorithm that takes advantage of the fact
        that both lists are sorted by internal document ID.
        
        Args:
            list1: First sorted list of internal document IDs.
            list2: Second sorted list of internal document IDs.
            
        Returns:
            A sorted list containing the intersection (documents in both lists).
        """
        result = []
        i, j = 0, 0
        
        while i < len(list1) and j < len(list2):
            if list1[i] == list2[j]:
                result.append(list1[i])
                i += 1
                j += 1
            elif list1[i] < list2[j]:
                i += 1
            else:
                j += 1
        
        return result
    
    def _union(self, list1: List[int], list2: List[int]) -> List[int]:
        """
        Compute the union of two sorted postings lists.
        
        Uses a two-pointer merge algorithm that takes advantage of the fact
        that both lists are sorted by internal document ID.
        
        Args:
            list1: First sorted list of internal document IDs.
            list2: Second sorted list of internal document IDs.
            
        Returns:
            A sorted list containing the union (documents in either list).
        """
        result = []
        i, j = 0, 0
        
        while i < len(list1) and j < len(list2):
            if list1[i] < list2[j]:
                result.append(list1[i])
                i += 1
            elif list1[i] > list2[j]:
                result.append(list2[j])
                j += 1
            else:
                # Both are equal, add once
                result.append(list1[i])
                i += 1
                j += 1
        
        # Add remaining elements
        while i < len(list1):
            result.append(list1[i])
            i += 1
        
        while j < len(list2):
            result.append(list2[j])
            j += 1
        
        return result
    
    def _complement(self, postings: List[int]) -> List[int]:
        """
        Compute the complement of a postings list.
        
        Returns all document IDs (0 to total_docs-1) that are NOT in the
        given postings list. Uses a merge-based algorithm on sorted lists.
        
        Args:
            postings: A sorted list of internal document IDs to complement.
            
        Returns:
            A sorted list containing all document IDs not in the input list.
        """
        result = []
        postings_idx = 0
        
        # Generate all document IDs from 0 to total_docs-1
        # and skip those that are in the postings list
        for doc_id in range(self.total_docs):
            # Advance postings_idx until we find a doc_id >= current doc_id
            while postings_idx < len(postings) and postings[postings_idx] < doc_id:
                postings_idx += 1
            
            # If we've reached the end or found a larger element, this doc_id is not in postings
            if postings_idx >= len(postings) or postings[postings_idx] != doc_id:
                result.append(doc_id)
        
        return result
    
    def _and_not(self, list1: List[int], list2: List[int]) -> List[int]:
        """
        Compute list1 AND NOT list2.
        
        Returns documents that are in list1 but not in list2.
        Uses a two-pointer algorithm on sorted lists.
        
        Args:
            list1: First sorted list of internal document IDs.
            list2: Second sorted list of internal document IDs (to exclude).
            
        Returns:
            A sorted list containing documents in list1 but not in list2.
        """
        result = []
        i, j = 0, 0
        
        while i < len(list1):
            # Advance j until list2[j] >= list1[i] or j reaches end
            while j < len(list2) and list2[j] < list1[i]:
                j += 1
            
            # If we've reached end of list2 or found a larger element,
            # list1[i] is not in list2, so include it
            if j >= len(list2) or list2[j] > list1[i]:
                result.append(list1[i])
            
            i += 1
        
        return result
    
    def _evaluate_rpn_query(self, query_tokens: List[str]) -> List[int]:
        """
        Evaluate a Boolean query in Reverse Polish Notation.
        
        The query is processed using a stack-based algorithm:
        - Terms are pushed as their postings lists
        - Operators (AND, OR, NOT) pop operands, compute results, and push back
        
        Args:
            query_tokens: List of tokens representing the RPN query.
            
        Returns:
            A sorted list of internal document IDs matching the query.
        """
        stack = []
        
        for token in query_tokens:
            token = token.strip().lower()
            
            if not token:
                continue
            
            if token == 'and':
                # Pop two operands, compute intersection, push result
                if len(stack) < 2:
                    raise ValueError("Invalid query: AND requires two operands")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._intersect(operand1, operand2)
                stack.append(result)
            
            elif token == 'or':
                # Pop two operands, compute union, push result
                if len(stack) < 2:
                    raise ValueError("Invalid query: OR requires two operands")
                operand2 = stack.pop()
                operand1 = stack.pop()
                result = self._union(operand1, operand2)
                stack.append(result)
            
            elif token == 'not':
                # NOT is treated as AND NOT
                # This means: (previous result) AND NOT (term)
                # So we need: pop the term to exclude, pop the previous result,
                # compute (previous result) AND NOT (term)
                if len(stack) < 2:
                    raise ValueError("Invalid query: NOT requires two operands (term and previous result)")
                term_to_exclude = stack.pop()  # The term we want to exclude
                previous_result = stack.pop()  # The previous query result
                result = self._and_not(previous_result, term_to_exclude)
                stack.append(result)
            
            else:
                # It's a term - get its postings list and push to stack
                postings = self.index.get_postings(token)
                stack.append(postings)
        
        # The final result should be on the stack
        if len(stack) != 1:
            raise ValueError(f"Invalid query: stack has {len(stack)} elements after evaluation")
        
        return stack[0]
    
    def process_query(self, query: str) -> List[str]:
        """
        Process a single Boolean query and return matching document IDs.
        
        Args:
            query: A Boolean query string in RPN format.
            
        Returns:
            A sorted list of original document IDs (DOCNO) matching the query.
        """
        # Tokenize the query
        query_tokens = query.strip().split()
        
        # Evaluate the query to get internal document IDs
        internal_ids = self._evaluate_rpn_query(query_tokens)
        
        # Convert internal IDs to original document IDs
        original_ids = []
        for internal_id in internal_ids:
            original_docno = self.index.get_original_docno(internal_id)
            if original_docno:
                original_ids.append(original_docno)
        
        # Sort by original document ID for consistent output
        original_ids.sort()
        
        return original_ids
    
    def process_queries_file(self, queries_file: str, output_file: str):
        """
        Process multiple queries from a file and write results to an output file.
        
        Each line in the queries file is treated as a separate query.
        Results are written to the output file, with each line containing
        space-separated original document IDs for the corresponding query.
        
        Args:
            queries_file: Path to the file containing queries (one per line).
            output_file: Path to the output file where results will be written.
        """
        results = []
        
        # Read and process each query
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                query = line.strip()
                if not query:
                    # Empty line - return empty result
                    results.append([])
                    continue
                
                try:
                    doc_ids = self.process_query(query)
                    results.append(doc_ids)
                    print(f"Query {line_num}: '{query}' -> {len(doc_ids)} documents")
                except Exception as e:
                    print(f"Error processing query {line_num} ('{query}'): {e}")
                    results.append([])
        
        # Write results to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_ids in results:
                # Write space-separated document IDs
                line = ' '.join(doc_ids)
                f.write(line + '\n')
        
        print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    # Example usage
    from inverted_index import InvertedIndex
    
    # Build the inverted index
    print("Building inverted index...")
    index = InvertedIndex()
    index.build_index("data")
    
    # Create Boolean retrieval system
    print("\nInitializing Boolean retrieval system...")
    retrieval = BooleanRetrieval(index)
    
    # Process queries
    print("\nProcessing queries...")
    retrieval.process_queries_file("queries/query.txt", "Part_2.txt")

