# Running on Google Colab - Complete Guide

## Method 1: Upload Files Directly to Colab (Easiest)

### Step 1: Open Colab and Upload Notebook

1. Go to https://colab.research.google.com/
2. Click **File → Upload Notebook**
3. Select `ex3/rag_system.ipynb`

### Step 2: Upload Data Files

1. In Colab, click the **folder icon** (📁) on the left sidebar
2. Click **Upload** button
3. Upload these files:
   - `train.csv` → Upload to `/content/data/` (create folder if needed)
   - `test.csv` → Upload to `/content/data/`
   - `example_prediction_file_for_reference.csv` (optional)

**OR** create the folder structure:
- Click **New Folder** → Name it `data`
- Upload `train.csv` and `test.csv` to the `data` folder

### Step 3: Run the Notebook

1. **Run the first cell** (Colab Setup) - this installs packages
2. **Run all cells sequentially** (Runtime → Run All)
3. The notebook will automatically detect Colab and adjust paths

### Step 4: Download Results

After processing completes:
1. Right-click on `predictions.csv` in the file browser
2. Click **Download**

---

## Method 2: Use Google Drive (For Large Files)

### Step 1: Upload to Google Drive

1. Create a folder in Google Drive (e.g., `rag-assignment`)
2. Upload these files to that folder:
   - `rag_system.ipynb`
   - `data/train.csv`
   - `data/test.csv`

### Step 2: Open from Drive

1. Go to https://colab.research.google.com/
2. Click **File → Open Notebook → Google Drive**
3. Navigate to your folder and open `rag_system.ipynb`

### Step 3: Update Paths

In the notebook, find the configuration cell and update:

```python
# Change this line:
DATA_DIR = Path("data")

# To:
DATA_DIR = Path("/content/drive/MyDrive/rag-assignment/data")
```

### Step 4: Run

1. Run the Colab setup cell (mounts Drive automatically)
2. Run all cells

---

## Colab-Specific Features

### GPU Acceleration

Colab provides free GPU access:
1. **Runtime → Change runtime type**
2. Select **GPU** (T4 or better)
3. The code will automatically use GPU if available

**Note**: The model uses `device_map="auto"` which automatically uses GPU if available.

### Runtime Limits

- **Free tier**: ~12 hours max runtime
- **Processing time**: ~1-3 hours for all 2032 questions
- **Solution**: Use checkpoint system - saves progress every 50 questions

### If Runtime Disconnects

1. The checkpoint file (`checkpoint.json`) saves your progress
2. Re-run the processing cell - it will automatically resume from checkpoint
3. Or manually load checkpoint and continue

---

## Quick Start Commands

### Option A: Direct Upload (Recommended)

```python
# 1. Upload notebook to Colab
# 2. Upload data files to /content/data/
# 3. Run all cells
```

### Option B: Google Drive

```python
# 1. Upload everything to Google Drive
# 2. Open notebook from Drive
# 3. Update DATA_DIR path in config cell
# 4. Run all cells
```

---

## Troubleshooting

### "File not found" error

**Solution**: Check file paths. In Colab:
- Files uploaded directly: Use `Path("data")` or `Path("ex3/data")`
- Files in Drive: Use `Path("/content/drive/MyDrive/your-folder/data")`

### "Module not found" error

**Solution**: Run the Colab setup cell first (installs all packages)

### "Out of memory" error

**Solution**: 
- Use Runtime → Change runtime type → High-RAM
- Or reduce batch size in code
- Or process in smaller chunks

### Runtime disconnected

**Solution**: 
- Checkpoint system saves progress
- Re-run processing cell to resume
- Or manually load checkpoint

### Slow processing

**Solution**:
- Enable GPU: Runtime → Change runtime type → GPU
- The model automatically uses GPU if available

---

## Expected Runtime

- **Index download**: First run only, ~10-30 minutes
- **Model download**: First run only, ~2-5 minutes  
- **Processing**: ~2-5 seconds per question
- **Total for 2032 questions**: ~1-3 hours

---

## File Structure in Colab

After setup, your Colab environment should have:

```
/content/
├── data/
│   ├── train.csv
│   └── test.csv
├── rag_system.ipynb
├── predictions.csv (generated)
└── checkpoint.json (auto-saved)
```

Or if using Drive:

```
/content/drive/MyDrive/rag-assignment/
├── data/
│   ├── train.csv
│   └── test.csv
├── rag_system.ipynb
└── predictions.csv (generated)
```

---

## Tips

1. **Start with small test**: Modify code to process first 10 questions to verify everything works
2. **Use GPU**: Enable GPU for faster processing
3. **Save frequently**: Checkpoint system saves every 50 questions
4. **Download results**: Don't forget to download `predictions.csv` before closing Colab

