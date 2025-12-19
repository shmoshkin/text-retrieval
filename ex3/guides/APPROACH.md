# RAG System Approach and Methodology

## System Overview

This document describes the implementation of a Retrieval-Augmented Generation (RAG) system for question answering. The system retrieves relevant Wikipedia passages using Pyserini and uses Llama-3.2-1B-Instruct to generate concise answers based on the retrieved context.

### Architecture

The system follows a standard RAG pipeline:

1. **Question Input**: Receive question from test dataset
2. **Retrieval**: Search Wikipedia corpus using Pyserini to find relevant passages
3. **Context Formation**: Combine top-k retrieved passages into context
4. **Prompt Engineering**: Format context and question into LLM prompt
5. **Answer Generation**: Use Llama-3.2-1B-Instruct to generate answer
6. **Answer Extraction**: Post-process LLM output to extract clean answer
7. **Output**: Format and save predictions in required CSV format

## Retrieval Strategy

### Ranking Models

The system supports two retrieval methods, with QLD as the primary method from the course:

#### 1. Query Likelihood Dirichlet (QLD) - PRIMARY METHOD

QLD is the main retrieval method covered in the course. It's a language modeling approach that ranks documents based on the probability of generating the query from the document language model, with Dirichlet smoothing to handle term frequency estimation.

**Implementation**:
- Uses Pyserini's `set_qld()` method (as shown in template and course)
- Configurable `mu` parameter (Dirichlet smoothing parameter)
- Default `mu = 1000` (from template)
- Experimented with `mu` values: 500, 1000, 2000, 3000

**Rationale**: QLD is the primary method taught in the course and used in the template. The Dirichlet smoothing parameter `mu` controls the influence of collection statistics vs. document-specific statistics. Higher mu values give more weight to collection statistics (smoother), while lower values give more weight to document-specific term frequencies.

#### 2. BM25 (Optional Alternative)

BM25 is a probabilistic ranking function that was mentioned in the course but not deeply covered.

**Implementation**:
- Uses Pyserini's `set_bm25()` method
- Configurable `k1` parameter (term frequency saturation)
- Configurable `b` parameter (length normalization)
- Default parameters: `k1=0.9`, `b=0.4`

**Note**: BM25 was mentioned briefly in the course material (PDF 10) but not as the primary method. It's included as an optional alternative for experimentation, but QLD is the main method taught and used in the template. The focus should be on QLD parameter tuning.

### Retrieval Parameters

**k (Number of Passages)**: 
- Default: 10 passages
- Experimented with: 5, 10, 15, 20
- Trade-off: More passages provide more context but may introduce noise
- Finding: k=10 provides good balance between coverage and relevance

**Context Length**:
- Default: 800 characters per passage
- Increased from template's 300 characters to preserve more information
- Can be set to 0 for full passages (limited by model token limit)

### Passage Selection

- Uses full passage content instead of truncated snippets (improved from template)
- Filters out documents that cannot be retrieved (error handling)
- Maintains passage order based on relevance scores
- No explicit reranking beyond initial retrieval scores

## Prompt Engineering

### System Prompt

The system prompt instructs the LLM to:
1. Use ONLY information from provided passages
2. Provide SHORT, DIRECT answers (typically 1-5 words)
3. Avoid explanations, citations, or additional context
4. Return "I don't know" if answer is not in passages
5. Extract answers directly without unnecessary paraphrasing

**Improvements over template**:
- More specific instructions about answer format
- Emphasis on conciseness
- Clear rules about what to avoid

### User Prompt

Format:
```
Based on the following passages, provide a concise answer to the question.

Passages:
[Passage 1]
[Passage 2]
...

Question: [question]

Answer:
```

**Improvements over template**:
- Better formatting with numbered passages
- Clear separation between context and question
- Explicit instruction for concise answers
- Fixed bug: uses `query` parameter instead of undefined `question` variable

### Answer Extraction

Post-processing steps to extract clean answers:
1. Remove common prefixes ("The answer is", "Answer:", etc.)
2. Extract first sentence if answer is embedded in explanation
3. Limit to 50 characters if answer is too long
4. Handle edge cases (empty responses, "I don't know")

## Experiments and Results

### Parameter Tuning

Experiments conducted on training set to optimize parameters:

1. **Retrieval Method Comparison**:
   - QLD (mu=1000): Baseline performance (primary method from course)
   - BM25 (k1=0.9, b=0.4): Optional alternative for experimentation
   - Note: Hybrid combination not covered in course, so not included

2. **k Value Experimentation**:
   - k=5: Faster but may miss relevant information
   - k=10: Good balance (chosen as default)
   - k=15-20: Diminishing returns, more noise

3. **QLD Mu Parameter**:
   - mu=500: More document-specific, may overfit
   - mu=1000: Baseline (good general performance)
   - mu=2000-3000: More collection-based, smoother estimates

4. **Context Length**:
   - 300 chars (template): Too restrictive
   - 800 chars (chosen): Better information retention
   - Full passages: Best but may exceed token limits

### Baseline Comparison

- **Template Baseline F1**: 11.62
- **Target**: Improve over baseline through optimizations
- **Key Improvements**:
  - Fixed bug in prompt creation (undefined variable)
  - Increased context length (300 → 800 chars)
  - Improved prompt engineering
  - Better answer extraction

## Technical Details

### Configuration Parameters

All parameters are configurable through the `Config` class (Python script) or configuration cell (notebook):

- **Retrieval**: `K`, `RETRIEVAL_METHOD`, `QLD_MU`, `BM25_K1`, `BM25_B`, `CONTEXT_LENGTH`
- **LLM**: `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P`, `DO_SAMPLE`
- **Processing**: `BATCH_SIZE`, `SAVE_CHECKPOINT_EVERY`, `RESUME_FROM_CHECKPOINT`

### Error Handling

- Handles missing Wikipedia index gracefully
- Handles model loading failures
- Handles malformed questions or empty retrievals
- Checkpoint system for long-running operations
- Resume capability if interrupted

### Performance Optimizations

- Checkpointing: Saves progress every N questions
- Progress tracking: Uses tqdm for visual feedback
- Efficient retrieval: Single pass through index
- Memory management: Processes one question at a time

## Code Organization

### File Structure

```
ex3/
├── rag_system.py              # Main Python script (MUST SUBMIT)
├── rag_system.ipynb           # Jupyter notebook version (MUST SUBMIT)
├── predictions.csv            # Final predictions output (MUST SUBMIT)
├── APPROACH.md                # This document (MUST SUBMIT)
├── data/
│   ├── train.csv              # Training data with answers
│   ├── test.csv               # Test data (questions only)
│   └── example_prediction_file_for_reference.csv
└── TemplateRAGAssignment_Upload.ipynb  # Template (reference only)
```

### Key Functions

**Retrieval Module**:
- `get_context_qld()`: QLD retrieval (primary method from course)
- `get_context_bm25()`: BM25 retrieval (optional alternative)
- `get_context()`: Main retrieval function

**Prompt Module**:
- `create_message()`: Create LLM prompt (fixed bug)
- `extract_answer()`: Post-process LLM output

**LLM Module**:
- `load_llm_pipeline()`: Load model
- `llm_answer()`: Generate answer using RAG

**Evaluation Module**:
- `normalize_answer()`: Text normalization
- `f1_score()`: Compute F1 score
- `score()`: Evaluate predictions

**Processing Module**:
- `process_test_questions()`: Main processing function
- `evaluate_on_train()`: Training set evaluation
- `save_checkpoint()` / `load_checkpoint()`: Checkpoint management

## How to Run

### Python Script

```bash
# Process test questions (default)
python ex3/rag_system.py --mode test

# Evaluate on training set
python ex3/rag_system.py --mode train

# Both
python ex3/rag_system.py --mode both

# With custom parameters
python ex3/rag_system.py --mode test --k 15 --method qld --qld-mu 2000
```

### Jupyter Notebook

1. Open `ex3/rag_system.ipynb`
2. Run cells sequentially
3. Adjust configuration parameters in cell 3
4. Run processing cells to generate predictions

## Key Findings and Insights

1. **Retrieval Method**: QLD with mu=1000 provides solid baseline (primary method from course). Focus on QLD parameter tuning (mu values) for optimization.

2. **Context Length**: Increasing from 300 to 800 characters significantly improves answer quality by preserving more relevant information.

3. **Prompt Engineering**: Clear, specific instructions about answer format are crucial for getting concise answers from the LLM.

4. **Answer Extraction**: Post-processing is essential to extract clean answers from LLM output, which often includes explanations.

5. **k Value**: k=10 provides good balance. Increasing beyond 10 shows diminishing returns and may introduce noise.

## Constraints Respected

- ✅ LLM: Used Llama-3.2-1B-Instruct (no version changes)
- ✅ Corpus: Used provided Wikipedia corpus (no modifications)
- ✅ Index: Used 'wikipedia-kilt-doc' pre-built index (no new index created)
- ✅ Prompts: Modified system and user prompts (allowed)

## Future Improvements

Potential areas for further optimization (within course scope):
1. QLD parameter tuning (mu values) - primary optimization focus
2. Query expansion techniques (mentioned in course but not required)
3. Adaptive k based on query complexity
4. Better answer extraction using post-processing
5. Prompt engineering improvements (allowed per assignment)

