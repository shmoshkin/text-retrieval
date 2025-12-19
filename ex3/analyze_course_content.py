#!/usr/bin/env python3
"""Extract detailed content from PDFs to understand course material."""
from pathlib import Path
from pypdf import PdfReader

pdf_dir = Path("ex3/pdfs")
pdf_files = sorted([f for f in pdf_dir.glob("*.pdf")], key=lambda x: int(x.stem))

# Key topics to search for
topics = {
    "QLD": ["query likelihood", "dirichlet", "QLD", "smoothing", "mu"],
    "BM25": ["BM25", "okapi", "k1", "b parameter"],
    "TF-IDF": ["TF-IDF", "tf-idf", "term frequency", "inverse document frequency"],
    "Vector Space": ["vector space", "cosine similarity", "VSM"],
    "Pyserini": ["pyserini", "lucene", "index"],
    "RAG": ["RAG", "retrieval augmented", "retrieval-augmented"],
    "Hybrid": ["hybrid", "combine", "fusion", "ensemble"],
    "Reranking": ["rerank", "re-rank", "second stage"],
    "Query Expansion": ["query expansion", "pseudo-relevance", "relevance feedback"]
}

print("=" * 80)
print("DETAILED COURSE CONTENT ANALYSIS")
print("=" * 80)

all_content = {}
for pdf_file in pdf_files:
    print(f"\n{'='*80}")
    print(f"Analyzing: {pdf_file.name}")
    print(f"{'='*80}")
    
    try:
        reader = PdfReader(str(pdf_file))
        all_text = ""
        for page in reader.pages:
            all_text += page.extract_text() + "\n"
        
        all_content[pdf_file.name] = all_text
        
        # Check which topics are covered
        found_topics = []
        for topic, keywords in topics.items():
            if any(kw.lower() in all_text.lower() for kw in keywords):
                found_topics.append(topic)
        
        if found_topics:
            print(f"Topics covered: {', '.join(found_topics)}")
        
        # Extract key sections
        lines = all_text.split('\n')
        key_sections = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["query likelihood", "dirichlet", "BM25", "okapi", "pyserini", "retrieval", "ranking"]):
                # Get context (previous and next lines)
                start = max(0, i-2)
                end = min(len(lines), i+5)
                context = '\n'.join(lines[start:end])
                if len(context) > 50:  # Only meaningful sections
                    key_sections.append(context[:500])
        
        if key_sections:
            print(f"\nFound {len(key_sections)} relevant sections")
            print("\n--- Sample Key Sections ---")
            for i, section in enumerate(key_sections[:3]):  # Show first 3
                print(f"\nSection {i+1}:")
                print(section)
                print("-" * 60)
                
    except Exception as e:
        print(f"Error: {e}")

# Summary of all topics
print("\n" + "=" * 80)
print("SUMMARY: TOPICS COVERED ACROSS ALL PDFs")
print("=" * 80)

for topic, keywords in topics.items():
    found_in = []
    for pdf_name, content in all_content.items():
        if any(kw.lower() in content.lower() for kw in keywords):
            found_in.append(pdf_name)
    if found_in:
        print(f"\n{topic}: Found in {len(found_in)} PDF(s)")
        print(f"  PDFs: {', '.join(found_in)}")

# Specific focus on PDF 10 (seems to have QLD and BM25)
if "10.pdf" in all_content:
    print("\n" + "=" * 80)
    print("DETAILED CONTENT FROM PDF 10 (QLD/BM25)")
    print("=" * 80)
    content_10 = all_content["10.pdf"]
    # Extract sections mentioning QLD or BM25
    lines = content_10.split('\n')
    relevant_lines = []
    for i, line in enumerate(lines):
        if any(term in line.lower() for term in ["qld", "bm25", "query likelihood", "dirichlet", "okapi", "mu", "k1", "b parameter"]):
            # Get surrounding context
            start = max(0, i-3)
            end = min(len(lines), i+8)
            context = '\n'.join(lines[start:end])
            relevant_lines.append(context)
    
    print(f"\nFound {len(relevant_lines)} relevant sections in PDF 10")
    for i, section in enumerate(relevant_lines[:10]):  # Show first 10
        print(f"\n--- Section {i+1} ---")
        print(section[:800])
        print("-" * 60)

