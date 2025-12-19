# Running on Google Colab

## Quick Setup Guide

### Step 1: Upload Files to Google Drive

1. **Create a folder** in your Google Drive (e.g., `rag-assignment`)
2. **Upload these files** to that folder:
   - `ex3/rag_system.ipynb` (the notebook)
   - `ex3/data/train.csv`
   - `ex3/data/test.csv`
   - `ex3/data/example_prediction_file_for_reference.csv`

### Step 2: Open in Colab

1. **Open Google Colab**: https://colab.research.google.com/
2. **Upload the notebook**:
   - File → Upload Notebook → Select `rag_system.ipynb`
   - OR: File → Open Notebook → Google Drive → Navigate to your folder

### Step 3: Mount Google Drive (if files are in Drive)

Add this cell at the beginning of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your folder
import os
os.chdir('/content/drive/MyDrive/rag-assignment')  # Change to your folder path
```

### Step 4: Install Dependencies

Add this cell after mounting drive:

```python
!pip install torch transformers huggingface_hub pandas tqdm pyserini
```

### Step 5: Set HuggingFace Token

Add this cell:

```python
import os
# Option 1: Set token directly (replace with your token)
os.environ['KAGGLE_API_TOKEN'] = "your_huggingface_token_here"

# Option 2: Or use the default token in the code
```

### Step 6: Update File Paths

In the notebook, update the data paths if needed:

```python
# If files are in current directory
DATA_DIR = Path("data")  # or "ex3/data" depending on your structure

# If files are in Drive
DATA_DIR = Path("/content/drive/MyDrive/rag-assignment/data")
```

## Complete Colab-Ready Notebook

I'll create a Colab-specific version that handles all these steps automatically.

