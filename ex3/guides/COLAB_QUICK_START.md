# Google Colab Quick Start

## 🚀 Fastest Way to Run

### Step 1: Upload to Colab

1. **Go to**: https://colab.research.google.com/
2. **File → Upload Notebook** → Select `ex3/rag_system.ipynb`

### Step 2: Upload Data Files

**Method A: Direct Upload (Easiest)**
1. Click **📁 folder icon** (left sidebar)
2. Click **📤 Upload** button
3. Create folder `data` (click "New Folder")
4. Upload `train.csv` and `test.csv` to the `data` folder

**Method B: Use Google Drive**
1. Upload files to Google Drive
2. In notebook, update `DATA_DIR` path in config cell

### Step 3: Run

1. **Run Cell 1** (Colab Setup) - installs packages and mounts Drive
2. **Run all cells** (Runtime → Run All)

That's it! 🎉

---

## 📋 What Each Cell Does

- **Cell 0**: Colab setup (install packages, mount Drive)
- **Cell 1**: Imports
- **Cell 2**: HuggingFace login
- **Cell 3**: Configuration (adjust parameters here)
- **Cell 4**: Load Wikipedia index (downloads ~10GB first time)
- **Cell 5**: Load data
- **Cell 6**: Load LLM model
- **Cell 7-9**: Define functions
- **Cell 10**: Process test questions (takes 1-3 hours)
- **Cell 11**: Save predictions

---

## ⚙️ Enable GPU (Faster Processing)

1. **Runtime → Change runtime type**
2. Select **GPU** (T4 or better)
3. The code automatically uses GPU if available

---

## 💾 Download Results

After processing:
1. Find `predictions.csv` in file browser
2. Right-click → **Download**

---

## 🔧 Troubleshooting

**"File not found"**
- Check file paths in config cell
- Verify files uploaded correctly

**"Module not found"**  
- Run Cell 0 (Colab setup) first

**Runtime disconnected**
- Checkpoint saves every 50 questions
- Re-run processing cell to resume

---

## ⏱️ Expected Time

- First run: ~30 min (downloads index + model)
- Processing: ~1-3 hours for all questions
- Progress bar shows status

---

## 📝 Notes

- Free Colab: ~12 hour runtime limit
- Checkpoint system: Saves progress automatically
- GPU recommended: Much faster processing

