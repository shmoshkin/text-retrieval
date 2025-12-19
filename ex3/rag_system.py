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
    DATA_DIR = Path("ex3/data")
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"
    PREDICTIONS_CSV = Path("ex3/predictions.csv")
    CHECKPOINT_FILE = Path("ex3/checkpoint.json")
    
    # HuggingFace token (set as environment variable or update here)
    HF_TOKEN = os.getenv("KAGGLE_API_TOKEN", "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")
    
    # Retrieval parameters
    K = 10  # Number of passages to retrieve
    RETRIEVAL_METHOD = "qld"  # "qld" (primary, from course) or "bm25" (optional)
    QLD_MU = 1000  # Dirichlet smoothing parameter for QLD (default from template)
    # Note: BM25 mentioned but not deeply covered in course - included as optional alternative
    BM25_K1 = 0.9  # BM25 k1 parameter (if using BM25)
    BM25_B = 0.4  # BM25 b parameter (if using BM25)
    CONTEXT_LENGTH = 800  # Max characters per passage (0 = no limit)
    
    # LLM parameters
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.6
    TOP_P = 0.9
    DO_SAMPLE = True
    
    # Processing
    BATCH_SIZE = 1  # Process one question at a time
    SAVE_CHECKPOINT_EVERY = 50  # Save checkpoint every N questions
    RESUME_FROM_CHECKPOINT = True  # Resume if checkpoint exists


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


# Note: Hybrid QLD+BM25 removed - not covered in course material
# Focus on QLD (primary method) and BM25 (optional alternative) separately


def get_context(searcher, query, k=10, retrieval_method="qld", config=None):
    """
    Retrieve relevant passages from Wikipedia index.
    
    Primary method: QLD (Query Likelihood Dirichlet) - main method from course
    Optional: BM25 - mentioned but not deeply covered, included for experimentation
    
    Args:
        searcher: Pyserini SimpleSearcher instance
        query: Question string
        k: Number of passages to retrieve
        retrieval_method: "qld" (primary) or "bm25" (optional)
        config: Config object with parameters
    
    Returns:
        List of context strings
    """
    if config is None:
        config = Config()
    
    # Retrieve hits based on method
    # Primary method: QLD (Query Likelihood Dirichlet) - main method from course
    if retrieval_method == "qld":
        hits = get_context_qld(searcher, query, k, mu=config.QLD_MU)
    elif retrieval_method == "bm25":
        # BM25 mentioned but not deeply covered - included as optional alternative
        hits = get_context_bm25(searcher, query, k, k1=config.BM25_K1, b=config.BM25_B)
    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}. Use 'qld' (primary) or 'bm25' (optional)")
    
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
    Create prompt messages for LLM.
    
    Fixed bug: uses 'query' parameter instead of undefined 'question' variable.
    Improved prompt for better answer extraction.
    """
    # Format contexts
    context_text = '\n\n'.join([f"Passage {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
    
    system_prompt = """You are a question-answering assistant. Your task is to provide concise, accurate answers based ONLY on the information provided in the passages below. 

Rules:
1. Use ONLY information from the provided passages
2. Provide a SHORT, DIRECT answer (typically 1-5 words)
3. Do NOT include explanations, citations, or additional context
4. If the answer is not in the passages, respond with "I don't know"
5. Extract the answer directly - do not paraphrase unnecessarily"""

    user_prompt = f"""Based on the following passages, provide a concise answer to the question.

Passages:
{context_text}

Question: {query}

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages


def extract_answer(text):
    """
    Extract clean answer from LLM output.
    Removes explanations, citations, and extra text.
    """
    if not text:
        return "I don't know"
    
    # Remove common prefixes
    text = text.strip()
    
    # Remove explanations that might come after the answer
    # Look for patterns like "The answer is X" or just extract first sentence
    sentences = text.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        # Remove question words if they appear
        first_sentence = re.sub(r'^(The answer is|Answer:|The answer:|It is|It\'s)', '', first_sentence, flags=re.IGNORECASE)
        first_sentence = first_sentence.strip()
        
        # If it's too long, it might be an explanation
        if len(first_sentence.split()) <= 10:
            return first_sentence
    
    # Fallback: return first 50 characters
    return text[:50].strip()


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
            model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
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
    parser.add_argument("--method", choices=["qld", "bm25"], default=Config.RETRIEVAL_METHOD,
                       help="Retrieval method: 'qld' (primary, from course) or 'bm25' (optional)")
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

