#!/usr/bin/env python3
"""Extract text from PDFs to analyze course content."""
import os
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
    except ImportError:
        print("No PDF library available")
        exit(1)

pdf_dir = Path("ex3/pdfs")
pdf_files = sorted([f for f in pdf_dir.glob("*.pdf")], key=lambda x: int(x.stem))

print("=" * 80)
print("COURSE MATERIAL SUMMARY")
print("=" * 80)

for pdf_file in pdf_files:
    print(f"\n{'='*80}")
    print(f"PDF: {pdf_file.name}")
    print(f"{'='*80}")
    
    try:
        reader = PdfReader(str(pdf_file))
        num_pages = len(reader.pages)
        print(f"Pages: {num_pages}")
        
        # Extract first page and last page for overview
        if num_pages > 0:
            first_page = reader.pages[0].extract_text()
            print("\n--- First Page (Title/Overview) ---")
            print(first_page[:1000])
            
            if num_pages > 1:
                last_page = reader.pages[-1].extract_text()
                print("\n--- Last Page (Summary/Conclusion) ---")
                print(last_page[:1000])
        
        # Extract all text for key terms
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
        
        # Look for key terms
        key_terms = ["QLD", "BM25", "retrieval", "ranking", "Pyserini", "RAG", 
                     "query likelihood", "Dirichlet", "hybrid", "reranking"]
        found_terms = []
        for term in key_terms:
            if term.lower() in all_text.lower():
                found_terms.append(term)
        
        if found_terms:
            print(f"\nKey terms found: {', '.join(found_terms)}")
            
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")

