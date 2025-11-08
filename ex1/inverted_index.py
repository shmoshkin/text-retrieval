"""
Inverted Index Implementation for AP Collection

This module implements an InvertedIndex class that builds an inverted index
from the AP corpus. The inverted index maps terms to lists of document IDs
that contain those terms, enabling efficient document retrieval.

The class uses successive integers (0, 1, 2, ...) as internal document IDs
for optimized query processing, while maintaining a mapping to the original
document IDs (DOCNO) for reference.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set


class InvertedIndex:
    """
    A class to build and manage an inverted index for the AP collection.
    
    The inverted index is a data structure that maps terms to the list of
    documents (by internal ID) that contain those terms. This enables efficient
    lookup of all documents containing a specific term.
    
    Attributes:
        index (Dict[str, List[int]]): The inverted index mapping terms to
            sorted lists of internal document IDs.
        doc_id_mapping (Dict[int, str]): Mapping from internal document ID
            to original document ID (DOCNO).
        doc_count (int): Total number of documents indexed.
    """
    
    def __init__(self):
        """
        Initialize an empty InvertedIndex.
        
        The index starts empty and is populated by calling the build_index method.
        """
        # Inverted index: term -> sorted list of internal document IDs
        self.index: Dict[str, List[int]] = defaultdict(list)
        
        # Mapping from internal document ID to original DOCNO
        self.doc_id_mapping: Dict[int, str] = {}
        
        # Counter for assigning internal document IDs
        self._next_internal_id = 0
        
        # Total number of documents indexed
        self.doc_count = 0
    
    def _parse_document(self, doc_element: ET.Element) -> tuple:
        """
        Parse a single XML document element and extract its content.
        
        Args:
            doc_element: An XML Element representing a <DOC> tag.
            
        Returns:
            A tuple (docno, text) where:
                - docno (str): The document ID from <DOCNO> tag
                - text (str): Concatenated text from all <TEXT> tags
        """
        docno = None
        text_parts = []
        
        # Extract DOCNO
        docno_elem = doc_element.find('DOCNO')
        if docno_elem is not None:
            docno = docno_elem.text.strip() if docno_elem.text else None
        
        # Extract all TEXT tags (documents may have multiple TEXT tags)
        text_elements = doc_element.findall('TEXT')
        for text_elem in text_elements:
            if text_elem.text:
                text_parts.append(text_elem.text.strip())
        
        # Concatenate all text parts
        full_text = ' '.join(text_parts)
        
        return docno, full_text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.
        
        Since the text is already preprocessed (lowercased and punctuation
        removed), we only need to split on whitespace.
        
        Args:
            text: The text string to tokenize.
            
        Returns:
            A list of tokens (terms).
        """
        if not text:
            return []
        
        # Split on whitespace and filter out empty strings
        tokens = [token for token in text.split() if token]
        return tokens
    
    def _add_document(self, docno: str, text: str):
        """
        Add a single document to the inverted index.
        
        This method assigns an internal document ID, tokenizes the text,
        and updates the inverted index with the document's terms.
        
        Args:
            docno: The original document ID (DOCNO).
            text: The document text to index.
        """
        # Assign internal document ID
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        
        # Store mapping from internal ID to original DOCNO
        self.doc_id_mapping[internal_id] = docno
        
        # Tokenize the text
        tokens = self._tokenize(text)
        
        # Track unique terms in this document to avoid duplicates in postings
        unique_terms = set(tokens)
        
        # Add document ID to postings list for each unique term
        for term in unique_terms:
            self.index[term].append(internal_id)
        
        self.doc_count += 1
    
    def _parse_xml_file(self, file_path: str):
        """
        Parse an XML file containing multiple documents.
        
        The file may contain multiple <DOC> elements without a root element.
        Each document is processed and added to the inverted index.
        
        Args:
            file_path: Path to the XML file to parse.
        """
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Wrap multiple DOC elements in a root element for parsing
            # This handles files that have multiple <DOC> elements without a root
            wrapped_content = f'<ROOT>{content}</ROOT>'
            
            # Parse the wrapped XML
            root = ET.fromstring(wrapped_content)
            
            # Find all DOC elements
            doc_elements = root.findall('DOC')
            
            # Process each document
            for doc_elem in doc_elements:
                docno, text = self._parse_document(doc_elem)
                if docno and text:
                    self._add_document(docno, text)
                    
        except ET.ParseError as e:
            print(f"Warning: Error parsing {file_path}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
    
    def build_index(self, data_dir: str):
        """
        Build the inverted index from the AP corpus.
        
        This method processes all files in the AP_Coll_Parsed_* directories
        within the specified data directory. It parses each XML file and
        builds the inverted index.
        
        After building, the postings lists are sorted for efficient query processing.
        
        Args:
            data_dir: Path to the directory containing AP_Coll_Parsed_* subdirectories.
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Find all AP_Coll_Parsed_* directories
        parsed_dirs = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d)) 
                      and d.startswith('AP_Coll_Parsed_')]
        
        parsed_dirs.sort()  # Process in order
        
        print(f"Found {len(parsed_dirs)} AP collection directories")
        
        # Process each AP_Coll_Parsed_* directory
        for parsed_dir in parsed_dirs:
            parsed_dir_path = os.path.join(data_dir, parsed_dir)
            print(f"Processing {parsed_dir}...")
            
            # Get all files in the directory
            files = [f for f in os.listdir(parsed_dir_path) 
                    if os.path.isfile(os.path.join(parsed_dir_path, f))]
            
            # Process each file
            for filename in sorted(files):
                file_path = os.path.join(parsed_dir_path, filename)
                self._parse_xml_file(file_path)
        
        # Sort all postings lists for efficient query processing
        print("Sorting postings lists...")
        for term in self.index:
            self.index[term].sort()
        
        print(f"Index built successfully. Indexed {self.doc_count} documents.")
        print(f"Index contains {len(self.index)} unique terms.")
    
    def get_postings(self, term: str) -> List[int]:
        """
        Get the postings list for a given term.
        
        Args:
            term: The term to look up.
            
        Returns:
            A sorted list of internal document IDs containing the term.
            Returns an empty list if the term is not in the index.
        """
        return self.index.get(term, [])
    
    def get_original_docno(self, internal_id: int) -> str:
        """
        Get the original document ID (DOCNO) for a given internal document ID.
        
        Args:
            internal_id: The internal document ID.
            
        Returns:
            The original document ID (DOCNO), or None if not found.
        """
        return self.doc_id_mapping.get(internal_id)
    
    def get_vocabulary_size(self) -> int:
        """
        Get the number of unique terms in the index.
        
        Returns:
            The size of the vocabulary (number of unique terms).
        """
        return len(self.index)
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents indexed.
        
        Returns:
            The total number of documents.
        """
        return self.doc_count


if __name__ == "__main__":
    # Example usage
    index = InvertedIndex()
    
    # Build index from AP corpus
    data_directory = "data"
    index.build_index(data_directory)
    
    # Example: Look up a term
    term = "president"
    postings = index.get_postings(term)
    print(f"\nTerm '{term}' appears in {len(postings)} documents")
    if postings:
        print(f"First 10 document IDs: {postings[:10]}")
        # Get original DOCNO for first document
        first_docno = index.get_original_docno(postings[0])
        print(f"First document original ID: {first_docno}")

