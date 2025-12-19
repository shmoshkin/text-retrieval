# How to Run the RAG System

## Prerequisites

### 1. Install Dependencies

Make sure you have all required packages installed:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Additional dependencies needed (if not already installed):
pip install torch transformers huggingface_hub pandas tqdm
```

### 2. HuggingFace Token

The system needs a HuggingFace token to access the Llama model. You can either:

**Option A: Set as environment variable (recommended)**
```bash
export KAGGLE_API_TOKEN="your_huggingface_token_here"
```

**Option B: The code will use a default token if not set**
- The default token is already in the code
- You can also modify `HF_TOKEN` in the Config class if needed

### 3. Wikipedia Index

The system will automatically download the Wikipedia KILT index on first run:
- Index name: `wikipedia-kilt-doc`
- Size: ~10GB (downloads automatically)
- First run will take time to download

## Running the System

### Option 1: Python Script (Recommended for Full Run)

#### Basic Usage

```bash
# Process all test questions and generate predictions
python3 ex3/rag_system.py --mode test
```

#### With Custom Parameters

```bash
# Use different QLD mu parameter
python3 ex3/rag_system.py --mode test --qld-mu 2000

# Use different number of passages
python3 ex3/rag_system.py --mode test --k 15

# Use BM25 instead of QLD (optional)
python3 ex3/rag_system.py --mode test --method bm25

# Evaluate on training set (for parameter tuning)
python3 ex3/rag_system.py --mode train

# Both test and train
python3 ex3/rag_system.py --mode both
```

#### Command Line Arguments

- `--mode`: `test` (generate predictions), `train` (evaluate), or `both`
- `--k`: Number of passages to retrieve (default: 10)
- `--method`: `qld` (primary, default) or `bm25` (optional)
- `--qld-mu`: QLD Dirichlet smoothing parameter (default: 1000)

#### Output

- Predictions saved to: `ex3/predictions.csv`
- Checkpoint saved to: `ex3/checkpoint.json` (allows resuming if interrupted)

### Option 2: Jupyter Notebook (Recommended for Development/Experimentation)

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
# or
jupyter lab
```

2. **Open the notebook**:
   - Navigate to `ex3/rag_system.ipynb`

3. **Run cells sequentially**:
   - Cell 1: Setup and imports
   - Cell 2: HuggingFace authentication
   - Cell 3: Configuration (adjust parameters here if needed)
   - Cell 4: Load Wikipedia index (first time will download)
   - Cell 5: Load data
   - Cell 6: Load LLM model
   - Cell 7: Define retrieval functions
   - Cell 8: Define prompt functions
   - Cell 9: Define answer generation function
   - Cell 10: Define evaluation functions
   - Cell 11: Process test questions (this will take time!)
   - Cell 12: Format and save predictions
   - Cell 13: (Optional) Evaluate on training set

4. **Adjust Parameters**:
   - Edit Cell 3 to change:
     - `K`: Number of passages (default: 10)
     - `RETRIEVAL_METHOD`: "qld" or "bm25"
     - `QLD_MU`: Dirichlet smoothing (default: 1000)
     - `CONTEXT_LENGTH`: Max chars per passage (default: 800)

## Expected Runtime

- **Index Download**: First run only, ~10-30 minutes depending on connection
- **Model Loading**: ~1-2 minutes
- **Processing**: ~2-5 seconds per question
  - For 2032 test questions: ~1-3 hours total
  - Progress bar will show status

## Checkpoint System

The system automatically saves checkpoints every 50 questions (configurable). If interrupted:
- Checkpoint file: `ex3/checkpoint.json`
- Resume automatically on next run
- To start fresh, delete the checkpoint file

## Troubleshooting

### Issue: "Unable to download pre-built index"
- **Solution**: Check internet connection. The index is ~10GB and downloads automatically.

### Issue: "CUDA out of memory" or slow processing
- **Solution**: The model uses `device_map="auto"` which should handle GPU/CPU automatically. If issues persist, you may need to reduce batch size or use CPU.

### Issue: "HF_TOKEN does not exist"
- **Solution**: Set the `KAGGLE_API_TOKEN` environment variable or update the token in the Config class.

### Issue: Missing predictions
- **Solution**: Check the checkpoint file. You can resume from where it stopped.

## Quick Start Example

```bash
# 1. Set token (if needed)
export KAGGLE_API_TOKEN="your_token_here"

# 2. Run with default settings (QLD, mu=1000, k=10)
python3 ex3/rag_system.py --mode test

# 3. Check output
cat ex3/predictions.csv | head -10
```

## Testing on Small Subset

To test quickly before running on all questions, you can modify the code temporarily:

```python
# In rag_system.py, modify process_test_questions():
# Change: for index, row in tqdm(df_test.iterrows(), ...)
# To: for index, row in tqdm(df_test.head(10).iterrows(), ...)  # Test on first 10
```

Or in the notebook, modify Cell 11:
```python
# Test on first 10 questions
for index, row in tqdm(df_test.head(10).iterrows(), total=10, desc="Processing questions"):
    ...
```

## Output Format

The predictions file (`ex3/predictions.csv`) will have:
- Column 1: `id` - Question ID
- Column 2: `prediction` - JSON array format: `["answer"]`

Example:
```csv
id,prediction
1,"[""Jamaican Patois""]"
2,"[""Tennessee""]"
```

