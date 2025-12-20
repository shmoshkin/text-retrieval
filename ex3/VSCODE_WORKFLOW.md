# VS Code + Colab Workflow

## Quick Start

### 1. Install VS Code Extensions

```bash
# In VS Code, install these extensions:
# - Python (ms-python.python)
# - Jupyter (ms-toolsai.jupyter)
# - Pylance (ms-python.vscode-pylance)
```

Or via command line:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.vscode-pylance
```

### 2. Open Project in VS Code

```bash
cd /Users/amitshmoshkin/personal/text-retrieval
code .
```

### 3. Edit Notebook in VS Code

- Open `ex3/rag_system.ipynb`
- VS Code will show notebook interface
- Edit cells, run individual cells for testing
- Use IntelliSense, debugging, etc.

### 4. Sync to Colab

**Option A: Manual Upload**

1. Save notebook in VS Code
2. Go to Colab
3. File → Upload Notebook → Select `rag_system.ipynb`

**Option B: Google Drive Sync**

```bash
# Run sync script
python3 ex3/sync_to_colab.py

# Then open from Drive in Colab
```

**Option C: Use Google Drive Desktop**

- Install Google Drive for Desktop
- Place project in synced folder
- Edit in VS Code
- Open from Drive in Colab

### 5. Run in Colab

- Open notebook from Drive/upload
- Run all cells
- Download results

---

## Recommended Development Workflow

### For Code Development:

1. **Edit `rag_system.py` in VS Code** (better IDE support)

   - Full IntelliSense
   - Debugging
   - Refactoring tools

2. **Test changes locally** (if you have GPU/local setup):

   ```bash
   python3 ex3/rag_system.py --mode train  # Test on small subset
   ```

3. **Convert to notebook** (if needed):

   - The notebook and script should stay in sync
   - Or use `jupytext` for automatic sync

4. **Upload to Colab for full run**:
   - Upload notebook
   - Run full processing

### For Notebook Development:

1. **Edit `rag_system.ipynb` in VS Code**

   - Use notebook interface
   - Run cells individually
   - Test small changes

2. **Sync to Colab**:
   - Upload or use Drive sync
   - Run full pipeline

---

## VS Code Features You Can Use

### 1. IntelliSense & Autocomplete

- Full Python autocomplete
- Import suggestions
- Parameter hints

### 2. Debugging

- Set breakpoints in cells
- Step through code
- Inspect variables

### 3. Git Integration

- Version control
- Commit changes
- See diffs

### 4. Terminal Integration

- Run commands in integrated terminal
- Test scripts locally

---

## Tips

1. **Use Python script for development**:

   - Better IDE support
   - Easier to debug
   - Better Git diffs

2. **Use Notebook for experimentation**:

   - Quick testing
   - Interactive development
   - Visual output

3. **Keep both in sync**:

   - Make changes in script
   - Test in notebook
   - Or use `jupytext` for automatic sync

4. **Use Colab for heavy lifting**:
   - GPU access
   - Long-running processes
   - Large data processing

---

## Quick Commands

```bash
# Open in VS Code
code ex3/rag_system.ipynb

# Sync to Drive (if script exists)
python3 ex3/sync_to_colab.py

# Test locally (small subset)
python3 ex3/rag_system.py --mode train
```

---

## Troubleshooting

**VS Code doesn't show notebook preview:**

- Install Jupyter extension
- Reload VS Code

**Can't run cells in VS Code:**

- Select Python interpreter (Cmd+Shift+P → "Python: Select Interpreter")
- Choose your venv Python

**Changes not syncing:**

- Save file in VS Code (Cmd+S)
- Manually upload to Colab/Drive



