# RAG System for Question Answering

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set HuggingFace token (optional, defaults to provided token):
```bash
export KAGGLE_API_TOKEN="your_token_here"
```

### Running the System

#### Option 1: Python Script

```bash
# Process test questions and generate predictions
python ex3/rag_system.py --mode test

# Evaluate on training set
python ex3/rag_system.py --mode train

# Both
python ex3/rag_system.py --mode both

# With custom parameters
python ex3/rag_system.py --mode test --k 15 --method hybrid --qld-mu 2000
```

#### Option 2: Jupyter Notebook

1. Open `ex3/rag_system.ipynb`
2. Run all cells sequentially
3. Adjust configuration in cell 3 if needed
4. Predictions will be saved to `ex3/predictions.csv`

## Output

- **Predictions**: `ex3/predictions.csv` (CSV with `id` and `prediction` columns)
- **Format**: Each prediction is a JSON array: `["answer"]`

## Configuration

Key parameters (adjustable in code/config):
- `K`: Number of passages to retrieve (default: 10)
- `RETRIEVAL_METHOD`: "qld", "bm25", or "hybrid" (default: "qld")
- `QLD_MU`: Dirichlet smoothing parameter (default: 1000)
- `CONTEXT_LENGTH`: Max characters per passage (default: 800)

## Files

- `rag_system.py`: Main Python script
- `rag_system.ipynb`: Jupyter notebook version
- `APPROACH.md`: Detailed methodology and approach
- `predictions.csv`: Final predictions (generated after running)

## Notes

- The system uses the pre-built 'wikipedia-kilt-doc' index (will download automatically)
- Processing all 2032 test questions may take several hours
- Checkpoint system allows resuming if interrupted
- See `APPROACH.md` for detailed methodology

