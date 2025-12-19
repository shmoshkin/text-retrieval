# Quick Guide: VS Code + Colab

## The Easiest Way

### 1. Edit in VS Code
```bash
# Open notebook in VS Code
code ex3/rag_system.ipynb
```

VS Code will show the notebook interface. Edit cells, use IntelliSense, debug, etc.

### 2. Sync to Colab

**Option A: Manual Upload (Simplest)**
1. Save in VS Code (Cmd+S / Ctrl+S)
2. Go to https://colab.research.google.com/
3. File → Upload Notebook
4. Select `ex3/rag_system.ipynb`

**Option B: Google Drive Sync**
```bash
# Run sync script
python3 ex3/sync_to_colab.py

# Then in Colab: File → Open Notebook → Google Drive
```

### 3. Run in Colab
- Run Cell 0 (Colab setup)
- Run all cells
- Download results

---

## Workflow Summary

```
VS Code (Edit) → Save → Upload to Colab → Run
```

That's it! 🎉

---

## Pro Tips

1. **Edit Python script** (`rag_system.py`) in VS Code for better IDE support
2. **Test small changes** in notebook locally first
3. **Run full pipeline** in Colab for GPU access
4. **Use Git** in VS Code to track changes

---

## VS Code Extensions Needed

- **Jupyter** (ms-toolsai.jupyter) - Edit notebooks
- **Python** (ms-python.python) - Python support

Install: `code --install-extension ms-toolsai.jupyter`

