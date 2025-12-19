# Quick Start - Running the RAG System

## Step 1: Install Dependencies

```bash
# Install required packages
pip install torch transformers huggingface_hub pandas tqdm pyserini

# Or install from requirements.txt (if it includes all dependencies)
pip install -r requirements.txt
```

**Note**: If you're using a virtual environment, activate it first:
```bash
source venv/bin/activate  # or your venv path
```

## Step 2: Set HuggingFace Token (Optional)

The code has a default token, but you can set your own:

```bash
export KAGGLE_API_TOKEN="your_huggingface_token_here"
```

Or the code will use the default token in `Config.HF_TOKEN`.

## Step 3: Run the System

### Quick Test (First 10 Questions)

To test quickly before running on all questions, you can temporarily modify the code or use a small subset.

**Option A: Run Python Script**
```bash
# Basic run (processes all test questions)
python3 ex3/rag_system.py --mode test

# With custom QLD mu parameter
python3 ex3/rag_system.py --mode test --qld-mu 2000

# Evaluate on training set first (recommended for tuning)
python3 ex3/rag_system.py --mode train
```

**Option B: Run Jupyter Notebook**
```bash
jupyter notebook ex3/rag_system.ipynb
```
Then run cells sequentially.

## Step 4: Check Output

After running, check the predictions:
```bash
# View first few predictions
head -10 ex3/predictions.csv

# Count total predictions
wc -l ex3/predictions.csv
```

## Expected Behavior

1. **First Run**: 
   - Downloads Wikipedia index (~10GB, takes 10-30 minutes)
   - Downloads Llama model (~2GB, takes a few minutes)
   - Then processes questions

2. **Subsequent Runs**:
   - Uses cached index and model
   - Processes questions immediately

3. **Progress**:
   - Shows progress bar
   - Saves checkpoint every 50 questions
   - Can resume if interrupted

## Troubleshooting

**"ModuleNotFoundError"**: Install missing packages with `pip install <package>`

**"Unable to download index"**: Check internet connection, index is ~10GB

**"CUDA out of memory"**: System uses CPU automatically, but you can reduce batch size if needed

**Want to test on small subset first?**: See HOW_TO_RUN.md for instructions

