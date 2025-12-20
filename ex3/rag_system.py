#!/usr/bin/env python3
"""
RAG System for Question Answering
Retrieves relevant Wikipedia passages and uses Llama-3.2-1B-Instruct to generate answers.
"""

import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import transformers
from huggingface_hub import login
from pyserini.search import SimpleSearcher
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration parameters for the RAG system."""
    
    # Data paths
    TRAIN_CSV = "./train.csv"
    TEST_CSV = "./test.csv"
    PREDICTIONS_CSV = "./predictions.csv"
    CHECKPOINT_FILE = "./checkpoint.json"
    
    # HuggingFace token (set as environment variable or update here)
    HF_TOKEN = os.getenv("KAGGLE_API_TOKEN", "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")
    
    # Retrieval parameters - OPTIMIZED
    K = 30  # Increased for better recall - more passages = better chance of finding answer
    RETRIEVAL_METHOD = "qld"  # "qld", "bm25", "rrf" (reciprocal rank fusion)
    QLD_MU = 1000  # Standard value - tune between 500-2000
    # Note: BM25 mentioned but not deeply covered in course - included as optional alternative
    BM25_K1 = 1.2  # BM25 k1 parameter (standard: 1.2)
    BM25_B = 0.75  # BM25 b parameter (standard: 0.75)
    CONTEXT_LENGTH = 0  # NO TRUNCATION! Keep full passages
    
    # Advanced retrieval options
    USE_RRF = False  # Reciprocal Rank Fusion - combine QLD and BM25 results
    RRF_K = 60  # Number of docs to retrieve from each method for RRF
    RRF_FINAL_K = 30  # Final number of docs after RRF fusion
    
    # LLM parameters - OPTIMIZED
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    MAX_NEW_TOKENS = 64  # Shorter answers reduce verbosity and extraction errors
    TEMPERATURE = 0.0  # Deterministic output for consistency
    TOP_P = 0.95  # Standard value for quality
    DO_SAMPLE = False  # Deterministic when temperature=0
    
    # Processing
    BATCH_SIZE = 1  # Process one question at a time
    SAVE_CHECKPOINT_EVERY = 50  # Save checkpoint every N questions
    RESUME_FROM_CHECKPOINT = True  # Resume if checkpoint exists
    
    # Debug options
    DEBUG_PRINT_CONTEXTS = False  # When True, print retrieved passages before sending to LLM


# ============================================================================
# Evaluation Functions (from template)
# ============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute token-level F1 score between prediction and a ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = 'id') -> float:
    """
    Computes average F1 score over all questions. Used for leaderboard ranking.
    """
    # Align dataframes and remove ID column
    gold = solution.set_index(row_id_column_name)
    pred = submission.set_index(row_id_column_name)

    f1_sum = 0.0
    count = 0

    for qid in gold.index:
        if qid not in pred.index:
            print(f"Missing prediction for question ID: {qid}")
            count += 1
            continue

        try:
            ground_truths = json.loads(gold.loc[qid, "answers"])
            if not isinstance(ground_truths, list):
                raise ValueError
        except Exception:
            raise Exception(f"Invalid format for answers at id {qid}: must be a JSON list of strings.")

        prediction = pred.loc[qid, "prediction"]
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)

        f1_sum += f1
        count += 1

    if count == 0:
        raise Exception("No matching question IDs between submission and solution.")

    return 100.0 * f1_sum / count


# ============================================================================
# Retrieval Functions
# ============================================================================

def get_context_qld(searcher, query, k, mu=1000):
    """Retrieve context using Query Likelihood Dirichlet (QLD) method."""
    searcher.set_qld(mu=mu)
    hits = searcher.search(query, k)
    return hits


def get_context_bm25(searcher, query, k, k1=0.9, b=0.4):
    """Retrieve context using BM25 method."""
    searcher.set_bm25(k1=k1, b=b)
    hits = searcher.search(query, k)
    return hits


def reciprocal_rank_fusion(hits_list, k=60, final_k=30):
    """
    Reciprocal Rank Fusion (RRF) to combine multiple retrieval results.
    
    Formula: RRF(d) = Σ(1 / (k + rank_i(d))) for each retrieval method i
    
    Args:
        hits_list: List of hit lists from different retrieval methods
        k: RRF constant (typically 60)
        final_k: Number of top documents to return after fusion
    
    Returns:
        List of fused hits sorted by RRF score
    """
    doc_scores = {}
    
    # Calculate RRF score for each document
    for method_hits in hits_list:
        for rank, hit in enumerate(method_hits, start=1):
            docid = hit.docid
            rrf_score = 1.0 / (k + rank)
            
            if docid not in doc_scores:
                doc_scores[docid] = {
                    'score': 0.0,
                    'hit': hit  # Keep first hit object we encounter
                }
            doc_scores[docid]['score'] += rrf_score
    
    # Sort by RRF score (descending)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Return top final_k hits
    fused_hits = []
    for docid, data in sorted_docs[:final_k]:
        # Create a simple hit-like object with docid
        class FusedHit:
            def __init__(self, docid, score):
                self.docid = docid
                self.score = score
        
        fused_hits.append(FusedHit(docid, data['score']))
    
    return fused_hits


def get_context(searcher, query, k=10, retrieval_method="qld", config=None):
    """
    Retrieve relevant passages from Wikipedia index.
    
    Methods supported:
    - "qld": Query Likelihood Dirichlet (primary method from course)
    - "bm25": BM25 ranking (optional alternative)
    - "rrf": Reciprocal Rank Fusion - combines QLD and BM25 results
    
    Args:
        searcher: Pyserini SimpleSearcher instance
        query: Question string
        k: Number of passages to retrieve
        retrieval_method: "qld", "bm25", or "rrf"
        config: Config object with parameters
    
    Returns:
        List of context strings
    """
    if config is None:
        config = Config()
    
    # Reciprocal Rank Fusion: combine QLD and BM25
    if retrieval_method == "rrf" or config.USE_RRF:
        # Retrieve from both methods
        qld_hits = get_context_qld(searcher, query, config.RRF_K, mu=config.QLD_MU)
        bm25_hits = get_context_bm25(searcher, query, config.RRF_K, k1=config.BM25_K1, b=config.BM25_B)
        
        # Fuse results using RRF
        hits = reciprocal_rank_fusion([qld_hits, bm25_hits], k=config.RRF_K, final_k=config.RRF_FINAL_K)
        
    elif retrieval_method == "qld":
        hits = get_context_qld(searcher, query, k, mu=config.QLD_MU)
    elif retrieval_method == "bm25":
        hits = get_context_bm25(searcher, query, k, k1=config.BM25_K1, b=config.BM25_B)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}. Use 'qld', 'bm25', or 'rrf'")
    
    # Extract passage text
    contexts = []
    for hit in hits:
        try:
            doc = searcher.doc(hit.docid)
            raw_json = doc.raw()
            data = json.loads(raw_json)
            contents = data['contents']
            
            # Clean and truncate if needed
            content = contents.replace('\n', ' ')
            if config.CONTEXT_LENGTH > 0 and len(content) > config.CONTEXT_LENGTH:
                content = content[:config.CONTEXT_LENGTH] + "..."
            
            contexts.append(content)
        except Exception as e:
            print(f"Warning: Could not retrieve document {hit.docid}: {e}")
            continue
    
    return contexts


# ============================================================================
# Prompt Engineering
# ============================================================================

def create_message(query, contexts):
    """
    Create prompt messages for LLM with few-shot examples.
    Uses simpler context format and teaches proper answer extraction.
    """
    # Simpler format: just join contexts with double newline (no passage numbering)
    context_text = '\n\n'.join(contexts)
    
    system_prompt = """Answer questions using ONLY the provided context. Extract the specific answer - don't repeat the question or use generic words. Give concrete names, places, or things.

CRITICAL: Extract just the answer entity, not full sentences or explanations.

Examples:

Context: "William Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language. His works include Romeo and Juliet, Hamlet, and Macbeth."
Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Context: "Paris is the capital and most populous city of France. It is located in the north-central part of the country."
Question: What is the capital of France?
Answer: Paris

Context: "North Port is a city in Sarasota County, Florida, United States. It is part of the North Port-Bradenton-Sarasota Metropolitan Statistical Area."
Question: Where is North Port Florida located?
Answer: Sarasota County

Context: "Joakim Noah plays for the Chicago Bulls in the NBA. He was drafted in 2007."
Question: Who does Joakim Noah play for?
Answer: Chicago Bulls

Context: "Iceland is a Nordic island country in the North Atlantic Ocean. Iceland belongs to the Nordic countries."
Question: What country does Iceland belong to?
Answer: Iceland

Context: "The Philippines gained independence from the United States of America in 1946 after World War II."
Question: Who did the Philippines gain independence from?
Answer: United States of America

Context: "You are currently in the North American Eastern Time Zone, which is UTC-5."
Question: What time zone am I in?
Answer: North American Eastern Time Zone

Context: "Shakespeare was born in Stratford-upon-Avon in 1564. He died in 1616 at the age of 52."
Question: What year was Shakespeare born?
Answer: 1564

Context: "Natalie Portman played Padmé Amidala in the Star Wars prequel trilogy. The character first appeared in Episode I: The Phantom Menace."
Question: What character did Natalie Portman play in Star Wars?
Answer: Padmé Amidala

Now answer this question:"""

    user_prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages


def print_contexts(query, contexts, max_chars=500, use_tqdm=False):
    """
    Print retrieved contexts for debugging and visibility.
    
    Args:
        query: The question being asked
        contexts: List of retrieved context strings
        max_chars: Maximum characters to show per passage (default: 500)
        use_tqdm: If True, use tqdm.write() instead of print() for better progress bar compatibility
    """
    # Use tqdm.write() if available and requested (for progress bar compatibility)
    if use_tqdm:
        try:
            from tqdm import tqdm
            write_func = tqdm.write
        except ImportError:
            write_func = print
    else:
        write_func = print
    
    write_func(f"\n{'='*80}")
    write_func(f"Query: {query}")
    write_func(f"Retrieved {len(contexts)} passages:")
    write_func(f"{'='*80}")
    for i, ctx in enumerate(contexts, 1):
        preview = ctx[:max_chars] + "..." if len(ctx) > max_chars else ctx
        write_func(f"\nPassage {i} ({len(ctx)} chars):")
        write_func(preview)
    write_func(f"{'='*80}\n")


def extract_answer(text):
    """
    Generic answer extraction that handles common LLM output patterns.
    Removes prefixes, explanations, and formatting artifacts.
    """
    if not text:
        return "I don't know"
    
    text = text.strip()
    
    # CRITICAL: Early filtering for common error patterns BEFORE processing
    # Filter out single digits (unless they're years like "1971")
    text_lower = text.strip().lower()
    if text_lower in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        # Check if it might be a year (4 digits) - if so, allow it
        if not re.match(r'^\d{4}$', text.strip()):
            return "I don't know"
    
    # Filter out common error words and meta-commentary
    invalid_answers = [
        "what", "who", "where", "when", "which", "how",
        "no answer", "no one", "none", "n/a", "na",
        "i don't know", "i do not know", "i cannot", "i can't",
        "the passages don't", "none of the passages"
    ]
    if text_lower in invalid_answers:
        return "I don't know"
    
    # Remove common LLM response prefixes
    prefixes = [
        r'^(The answer is|Answer:|The answer:|It is|It\'s|According to|Based on|From the passage|The passage states|I can see that|Looking at)',
        r'^(None of the passages|The passages don\'t|I cannot|I don\'t know|I do not know)',
    ]
    
    for pattern in prefixes:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    
    # Check again after prefix removal (might have been hidden by prefix)
    text_lower_after = text.strip().lower()
    if text_lower_after in invalid_answers or text_lower_after in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        if not re.match(r'^\d{4}$', text.strip()):
            return "I don't know"
    
    # Remove passage references (formatting artifacts)
    text = re.sub(r'\[Passage\s+\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Extract quoted text if present (often the most accurate)
    quoted_match = re.search(r'"([^"]+)"', text)
    if quoted_match:
        answer = quoted_match.group(1).strip()
        if answer:
            return answer
    
    # Remove trailing explanations (common pattern: "Answer. This is because...")
    # Look for explanation indicators
    explanation_patterns = [
        r'\s+(because|since|as|which|that|who|where|when|plays for|is from|was|were|did|does|do)\s+.*$',
        r'\s+\([^)]+\)\s*$',  # Parenthetical explanations
    ]
    
    for pattern in explanation_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    
    # Normalize whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    
    # IMPROVED: Detect and extract entities from verbose answers
    # Pattern 1: "X is Y" or "X was Y" → extract Y
    pattern1 = re.search(r'\b(is|was|are|were)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern1:
        entity = pattern1.group(2).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 2: "X belongs to Y" → extract Y
    pattern2 = re.search(r'belongs\s+to\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern2:
        entity = pattern2.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 3: "X gained independence from Y" → extract Y
    pattern3 = re.search(r'gained\s+independence\s+from\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern3:
        entity = pattern3.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 4: "I am in X" or "I am from X" → extract X
    pattern4 = re.search(r'I\s+am\s+(?:in|from)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern4:
        entity = pattern4.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 5: "X is located in Y" → extract Y
    pattern5 = re.search(r'is\s+located\s+in\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern5:
        entity = pattern5.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 6: "X is named Y" or "X is known as Y" → extract Y
    pattern6 = re.search(r'is\s+(?:named|known\s+as)\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern6:
        entity = pattern6.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Pattern 7: "X plays for Y" → extract Y (for "who does X play for" questions)
    pattern7 = re.search(r'plays?\s+for\s+(.+?)(?:\.|$)', text, re.IGNORECASE)
    if pattern7:
        entity = pattern7.group(1).strip().rstrip('.,;:')
        words = entity.split()
        if 1 <= len(words) <= 15:
            return entity.strip('"\'')
    
    # Split on sentence boundaries and take first sentence if reasonable
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        first_sentence = sentences[0].strip().rstrip('.,;:')
        words = first_sentence.split()
        
        # If first sentence is concise (likely the answer), use it
        if 1 <= len(words) <= 20:
            return first_sentence.strip('"\'')
    
    # IMPROVED: Better repetition detection - handle duplicate words/phrases
    # Split by whitespace and detect repeated phrases
    words_list = text.split()
    if len(words_list) > 1:
        # Check for immediate repetition (e.g., "Honey Honey" or "Kids Kids")
        unique_words = []
        prev_word = None
        for word in words_list:
            word_clean = word.strip('.,;:').lower()
            if word_clean != prev_word:  # Only add if different from previous
                unique_words.append(word)
                prev_word = word_clean
        if len(unique_words) < len(words_list) * 0.7:  # Significant reduction means repetition
            text = ' '.join(unique_words)
    
    # Handle repetition (common LLM error: repeating same word/phrase)
    lines = text.split('\n')
    unique_lines = []
    seen = set()
    for line in lines:
        line_clean = line.strip().lower()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            unique_lines.append(line.strip())
    
    if len(unique_lines) == 1:
        # Single unique line - likely the answer
        answer = unique_lines[0]
        words = answer.split()
        if 1 <= len(words) <= 20:
            return answer.strip('"\'.,;:')
    elif len(unique_lines) > 1:
        # Multiple lines - take first reasonable one
        for line in unique_lines:
            words = line.split()
            if 1 <= len(words) <= 15:
                return line.strip('"\'.,;:')
    
    # Fallback: return reasonable length text
    words = text.split()
    if len(words) <= 25:
        return ' '.join(words).strip('"\'.,;:')
    
    # Last resort: truncate at word boundary
    truncated = text[:100]
    last_space = truncated.rfind(' ')
    if last_space > 20:
        return truncated[:last_space].strip('"\'.,;:')
    
    return text[:100].strip()


def post_process_answer(answer):
    """
    Generic post-processing to clean extracted answers.
    Removes common artifacts and normalizes format.
    """
    if not answer:
        return "I don't know"
    
    answer = answer.strip()
    
    # Filter obviously invalid single-character or placeholder answers
    invalid_answers = ["none", "n/a", "na", "", "what", "who", "where", "when", "which", "how",
                      "no answer", "no one", "i don't know", "i do not know", "i cannot"]
    if answer.lower() in invalid_answers:
        return "I don't know"
    
    # Filter single digits (unless it's a year)
    if answer.strip() in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
        if not re.match(r'^\d{4}$', answer.strip()):
            return "I don't know"
    
    # Remove any remaining passage references
    answer = re.sub(r'\[Passage\s+\d+\]', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\[.*?\]', '', answer)
    
    # Remove trailing explanations that might have been missed
    answer = re.sub(r'\s+(because|since|as|which|that|plays for|is from|was|were)\s+.*$', '', answer, flags=re.IGNORECASE)
    
    # Remove parentheticals
    answer = re.sub(r'\([^)]*\)', '', answer)
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    answer = answer.replace('\n', ' ')
    
    # Remove trailing punctuation (but preserve if part of answer)
    answer = answer.rstrip('.,;:')
    
    return answer.strip()


# ============================================================================
# LLM Functions
# ============================================================================

def load_llm_pipeline(config=None):
    """Load Llama-3.2-1B-Instruct model and create pipeline."""
    if config is None:
        config = Config()
    
    print("Loading LLM model...")
    try:
        # Login to HuggingFace
        login(config.HF_TOKEN)
        
        # Load pipeline
        pipeline = transformers.pipeline(
            "text-generation",
            model=config.MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )   


        
        print("LLM model loaded successfully.")
        return pipeline
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise


def llm_answer(pipeline, searcher, query, config=None):
    """
    Generate answer using RAG pipeline.
    
    Args:
        pipeline: Transformers pipeline
        searcher: Pyserini searcher
        query: Question string
        config: Config object
    
    Returns:
        Answer string
    """
    if config is None:
        config = Config()
    
    try:
        # Retrieve context
        contexts = get_context(searcher, query, k=config.K, 
                             retrieval_method=config.RETRIEVAL_METHOD, 
                             config=config)
        
        if not contexts:
            return "I don't know"
        
        # Print contexts for debugging if enabled
        if config.DEBUG_PRINT_CONTEXTS:
            # Try to use tqdm.write if available (for progress bar compatibility)
            try:
                from tqdm import tqdm
                print_contexts(query, contexts, use_tqdm=True)
            except ImportError:
                print_contexts(query, contexts, use_tqdm=False)
        
        # Create prompt
        messages = create_message(query, contexts)
        
        # Get terminators
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        # Generate answer
        outputs = pipeline(
            messages,
            max_new_tokens=config.MAX_NEW_TOKENS,
            eos_token_id=terminators,
            do_sample=config.DO_SAMPLE,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )
        
        # Extract answer
        generated_text = outputs[0]["generated_text"][-1].get('content', '')
        answer = extract_answer(generated_text)
        answer = post_process_answer(answer)  # Final cleanup
        
        return answer
        
    except Exception as e:
        print(f"Error generating answer for query '{query}': {e}")
        return "I don't know"


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(predictions, processed_ids):
    """Save checkpoint to resume later."""
    checkpoint = {
        "predictions": predictions,
        "processed_ids": processed_ids
    }
    with open(Config.CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint saved: {len(predictions)} predictions")


def load_checkpoint():
    """Load checkpoint if exists."""
    if Config.CHECKPOINT_FILE.exists() and Config.RESUME_FROM_CHECKPOINT:
        try:
            with open(Config.CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            print(f"Checkpoint loaded: {len(checkpoint['predictions'])} predictions")
            return checkpoint["predictions"], set(checkpoint["processed_ids"])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return {}, set()


# ============================================================================
# Main Processing
# ============================================================================

def process_test_questions(config=None):
    """Process all test questions and generate predictions."""
    if config is None:
        config = Config()
    
    print("=" * 80)
    print("RAG System - Processing Test Questions")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading test data from {config.TEST_CSV}...")
    df_test = pd.read_csv(config.TEST_CSV)
    print(f"Loaded {len(df_test)} test questions")
    
    # Initialize searcher
    print("\nInitializing Pyserini searcher...")
    try:
        searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
        print("Searcher initialized successfully")
    except Exception as e:
        print(f"Error initializing searcher: {e}")
        print("Make sure the Wikipedia KILT index is available")
        raise
    
    # Load LLM
    pipeline = load_llm_pipeline(config)
    
    # Load checkpoint if exists
    predictions, processed_ids = load_checkpoint()
    
    # Process questions
    print(f"\nProcessing questions (method: {config.RETRIEVAL_METHOD}, k={config.K})...")
    
    for index, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing"):
        qid = row['id']
        question = row['question']
        
        # Skip if already processed
        if qid in processed_ids:
            continue
        
        # Generate answer
        answer = llm_answer(pipeline, searcher, question, config)
        predictions[qid] = answer
        
        # Save checkpoint periodically
        if (len(predictions) % config.SAVE_CHECKPOINT_EVERY == 0):
            save_checkpoint(predictions, list(processed_ids | {qid}))
        
        processed_ids.add(qid)
    
    # Final checkpoint save
    save_checkpoint(predictions, list(processed_ids))
    
    # Format and save predictions
    print("\nFormatting predictions...")
    df_prediction = pd.DataFrame(list(predictions.items()), columns=['id', 'prediction'])
    df_prediction = df_prediction.sort_values('id')
    df_prediction["prediction"] = df_prediction["prediction"].apply(
        lambda x: json.dumps([x], ensure_ascii=False)
    )
    
    print(f"\nSaving predictions to {config.PREDICTIONS_CSV}...")
    df_prediction.to_csv(config.PREDICTIONS_CSV, index=False)
    print(f"Saved {len(df_prediction)} predictions")
    
    return df_prediction


def evaluate_on_train(config=None):
    """Evaluate system on training set to compute F1 score."""
    if config is None:
        config = Config()
    
    print("=" * 80)
    print("Evaluating on Training Set")
    print("=" * 80)
    
    # Load training data
    print(f"\nLoading training data from {config.TRAIN_CSV}...")
    df_train = pd.read_csv(config.TRAIN_CSV, converters={"answers": json.loads})
    print(f"Loaded {len(df_train)} training questions")
    
    # Initialize searcher
    print("\nInitializing Pyserini searcher...")
    searcher = SimpleSearcher.from_prebuilt_index('wikipedia-kilt-doc')
    
    # Load LLM
    pipeline = load_llm_pipeline(config)
    
    # Process questions (sample for faster evaluation)
    print(f"\nProcessing training questions (method: {config.RETRIEVAL_METHOD}, k={config.K})...")
    predictions = {}
    
    # Process all or sample
    sample_size = len(df_train)  # Process all for full evaluation
    for index, row in tqdm(df_train.head(sample_size).iterrows(), total=sample_size, desc="Processing"):
        qid = row['id']
        question = row['question']
        answer = llm_answer(pipeline, searcher, question, config)
        predictions[qid] = answer
    
    # Format predictions
    df_pred = pd.DataFrame(list(predictions.items()), columns=['id', 'prediction'])
    df_pred = df_pred.sort_values('id')
    df_pred["prediction"] = df_pred["prediction"].apply(
        lambda x: json.dumps([x], ensure_ascii=False)
    )
    
    # Format ground truth
    df_gold = df_train.copy()
    df_gold["answers"] = df_gold["answers"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    
    # Evaluate
    print("\nComputing F1 score...")
    f1 = score(df_gold, df_pred)
    print(f"\nF1 Score on training set: {f1:.2f}")
    print(f"Baseline F1: 11.62")
    print(f"Improvement: {f1 - 11.62:.2f} points")
    
    return f1, df_pred


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System for Question Answering")
    parser.add_argument("--mode", choices=["test", "train", "both"], default="test",
                       help="Mode: test (generate predictions), train (evaluate), or both")
    parser.add_argument("--k", type=int, default=Config.K,
                       help="Number of passages to retrieve")
    parser.add_argument("--method", choices=["qld", "bm25", "rrf"], default=Config.RETRIEVAL_METHOD,
                       help="Retrieval method: 'qld' (primary), 'bm25' (optional), or 'rrf' (fusion)")
    parser.add_argument("--qld-mu", type=float, default=Config.QLD_MU,
                       help="QLD mu parameter")
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.K = args.k
    config.RETRIEVAL_METHOD = args.method
    config.QLD_MU = args.qld_mu
    
    if args.mode in ["test", "both"]:
        process_test_questions(config)
    
    if args.mode in ["train", "both"]:
        evaluate_on_train(config)

