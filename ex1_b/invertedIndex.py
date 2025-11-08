import heapq
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

import xml.etree.ElementTree as ET
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

@dataclass
class Document:
    # Represents a document in the corpus with an internal and external ID.
    internal_id: int
    external_id: str

    def __repr__(self):
        return f"{self.internal_id} ({self.external_id})"


PostingList = list[Document]


class InvertedIndex:
    def __init__(self, data_dir: str):
        """Initialize the InvertedIndex from files in a given data_dir.

        :param data_dir: Path to the directory containing the data files.
        """
        files: list[str] = self._scan_files(data_dir=data_dir)
        inverted_index, all_docs = self._create_inverted_index_from_files(files=files)
        self.inverted_index: dict[str, PostingList] = inverted_index
        self.all_docs: list[Document] = all_docs

    def __getitem__(self, word: str) -> PostingList:
        return self.inverted_index.get(word)

    @staticmethod
    def _scan_files(data_dir: str) -> list[str]:
        """Scan the directory and return a list of all valid file paths.

        :param data_dir: Directory path to scan.
        :return: A list of file paths.
        """
        return [path for path in Path(data_dir).rglob("*") if path.is_file() and '.DS_Store' not in str(path)]

    def _create_inverted_index_from_files(self, files: list[str]) -> tuple[dict[str, PostingList], list[Document]]:
        """Process the files and create the inverted index.

        :param files: List of file paths to process.
        :return: A dictionary representing the inverted index, and a list of all documents.
        """
        inverted_index: defaultdict[str, PostingList] = defaultdict(PostingList)
        internal_id = 1  # incrementing int ID for each document.
        all_docs: list[Document] = []

        # Process each file and extract documents.
        for file_path in tqdm(files, total=len(files)):
            documents_in_file: Iterator[dict] = filter(None, self._extract_documents_from_file(file_path=file_path))
            for document in documents_in_file:
                external_id = document["external_id"]
                text = document["text"]
                doc = Document(internal_id=internal_id, external_id=external_id)
                words = set(text.split())  # Use a set to avoid duplicate words in the same document.
                all_docs.append(doc)

                for word in words:
                    inverted_index[word].append(doc)

                internal_id += 1

        return inverted_index, all_docs

    def get_top_and_bottom_frequencies(self, top_n: int = 10) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Retrieve the top and bottom words by document frequency.

        :param top_n: Number of words to retrieve for each category.
        :return: A tuple of two lists (top_n_highest, top_n_lowest), each containing tuples of (word, frequency).
        """
        top_highest = heapq.nlargest(top_n, self.doc_frequencies.items(), key=lambda x: x[1])
        top_lowest = heapq.nsmallest(top_n, self.doc_frequencies.items(), key=lambda x: x[1])

        return top_highest, top_lowest

    @cached_property
    def doc_frequencies(self) -> dict[str, int]:
        """Compute and cache document frequencies for all words.

        :return: A dictionary with words as keys and their document frequencies as values.
        """
        return {word: len(postings) for word, postings in self.inverted_index.items()}

    @staticmethod
    def _extract_documents_from_file(file_path: str) -> Iterator[dict]:
        """Extract documents from a given XML file.

        :param file_path: Path to the XML file.
        :return: An iterator over the extracted documents, each represented as a dictionary with external_id and text.
        """
        try:
            with open(file_path, "r", encoding='utf-8', errors='ignore') as file:
                content = file.read()
        except Exception as e:
            print(rf"Couldn't read file {file_path}: {e}")
            raise

        # Wrap content in root element if needed (for files with multiple DOC elements)
        # Handle XML parsing errors gracefully
        try:
            wrapped_content = f'<ROOT>{content}</ROOT>'
            root = ET.fromstring(wrapped_content)
            docs = root.findall("DOC")
        except ET.ParseError:
            # If wrapping fails, try to parse individual DOC elements
            # This handles files with malformed XML
            docs = []
            # Try to find DOC elements using string parsing as fallback
            import re
            doc_pattern = r'<DOC>(.*?)</DOC>'
            doc_matches = re.finditer(doc_pattern, content, re.DOTALL)
            for match in doc_matches:
                doc_content = match.group(1)
                try:
                    doc_elem = ET.fromstring(f'<DOC>{doc_content}</DOC>')
                    docs.append(doc_elem)
                except:
                    continue
            if not docs:
                # If all else fails, return empty
                return

        for doc in docs:
            docno_elem = doc.find("DOCNO")
            external_id = docno_elem.text.strip() if docno_elem is not None and docno_elem.text else ""
            
            # Find all TEXT tags (documents may have multiple TEXT tags)
            text_elements = doc.findall("TEXT")
            all_text_elements = []
            for text_elem in text_elements:
                if text_elem.text:
                    all_text_elements.append(text_elem.text.strip())

            if all_text_elements and external_id:
                # Concatenate all text elements to form the document content.
                content = " ".join(all_text_elements)
                document = {
                    "external_id": external_id,
                    "text": content
                }
                yield document

    def get_posting_list(self, word: str) -> PostingList:
        return self.inverted_index.get(word)


def execute_and_write_to_file():
    """Create an inverted index, retrieve the answers required for Part_3 and write the results to a file."""
    import os
    
    # Find data directory - check multiple possible locations
    data_dir = None
    possible_data_dirs = ["data", "../ex1/data", "ex1/data", "../data"]
    for d in possible_data_dirs:
        if os.path.exists(d):
            data_dir = d
            break
    
    if data_dir is None:
        raise FileNotFoundError("Could not find data directory. Tried: " + ", ".join(possible_data_dirs))
    
    print(f"Using data directory: {data_dir}")
    index = InvertedIndex(data_dir)

    top_highest, top_lowest = index.get_top_and_bottom_frequencies(top_n=10)
    output_file = "Part_3.txt"

    with open(output_file, "w") as file:
        file.write("Top 10 Words with Highest Document Frequency:\n")
        for word, freq in top_highest:
            file.write(f"{word}: {freq}\n")

        file.write("\nTop 10 Words with Lowest Document Frequency:\n")
        for word, freq in top_lowest:
            file.write(f"{word}: {freq}\n")
    print(f"Results written to {output_file}")


if __name__ == '__main__':
    execute_and_write_to_file()