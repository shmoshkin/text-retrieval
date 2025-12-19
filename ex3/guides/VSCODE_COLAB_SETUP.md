# Using VS Code with Google Colab

You can edit your notebook in VS Code and run it on Colab. Here are the best approaches:

## Method 1: Edit in VS Code, Run in Colab (Recommended)

### Setup

1. **Install VS Code Jupyter Extension**:
   - Open VS Code
   - Install "Jupyter" extension (by Microsoft)
   - This allows you to edit `.ipynb` files in VS Code

2. **Edit Notebook Locally**:
   - Open `ex3/rag_system.ipynb` in VS Code
   - Edit code cells as needed
   - VS Code will show notebook preview

3. **Sync to Google Drive**:
   - Upload edited notebook to Google Drive
   - Or use Google Drive sync tool

4. **Run in Colab**:
   - Open notebook from Drive in Colab
   - Run cells there

### Workflow

```
VS Code (Edit) → Google Drive (Sync) → Colab (Run)
```

---

## Method 2: VS Code with Google Drive Sync

### Setup Google Drive Sync

1. **Install Google Drive Desktop** (or use rclone):
   ```bash
   # Option A: Google Drive for Desktop
   # Download from: https://www.google.com/drive/download/
   
   # Option B: Use rclone (command line)
   brew install rclone  # macOS
   rclone config  # Follow setup wizard
   ```

2. **Sync Your Project Folder**:
   - Set up sync between local folder and Google Drive
   - Edit in VS Code locally
   - Changes automatically sync to Drive
   - Open in Colab from Drive

### Workflow

```
VS Code (Edit) → Auto-sync to Drive → Colab (Run from Drive)
```

---

## Method 3: VS Code Remote Development (Advanced)

### Using Colab's Local Runtime (If Available)

1. **Set up Colab Local Runtime**:
   - This allows Colab to use your local machine
   - Requires Colab Pro/Pro+ subscription
   - More complex setup

2. **Connect VS Code**:
   - Use VS Code's remote development features
   - Connect to local runtime that Colab uses

**Note**: This is more complex and may not be worth it for this use case.

---

## Method 4: Hybrid Approach (Best for Development)

### Edit Python Script in VS Code, Convert to Notebook

1. **Edit `rag_system.py` in VS Code**:
   - Use full VS Code features (IntelliSense, debugging, etc.)
   - Make changes to the Python script

2. **Convert to Notebook** (if needed):
   - Use `jupytext` to sync `.py` and `.ipynb` files
   ```bash
   pip install jupytext
   jupytext --set-formats ipynb,py rag_system.py
   ```

3. **Upload to Colab**:
   - Upload the notebook version
   - Run in Colab

---

## Recommended Workflow

### For Development:

1. **Edit locally in VS Code**:
   - Use `rag_system.py` for development (better IDE support)
   - Use `rag_system.ipynb` for testing small changes

2. **Test locally first** (if possible):
   - Run small tests locally
   - Verify code works

3. **Upload to Colab for full run**:
   - Upload notebook to Colab
   - Run full processing there

### Quick Sync Script

Create a simple script to sync files:

```python
# sync_to_drive.py
import shutil
from pathlib import Path

# Copy notebook to Drive folder
notebook = Path("ex3/rag_system.ipynb")
drive_folder = Path("/path/to/Google Drive/rag-assignment/")

if drive_folder.exists():
    shutil.copy(notebook, drive_folder / "rag_system.ipynb")
    print("✅ Synced to Drive")
else:
    print("⚠️ Drive folder not found")
```

---

## VS Code Extensions to Install

1. **Jupyter** (Microsoft) - Edit notebooks
2. **Python** (Microsoft) - Python support
3. **Pylance** (Microsoft) - Better IntelliSense
4. **GitLens** (optional) - Git integration

---

## Tips

1. **Use VS Code for**:
   - Code editing and refactoring
   - Debugging
   - Git version control
   - Better autocomplete

2. **Use Colab for**:
   - Running full pipeline (GPU access)
   - Processing large datasets
   - Sharing results

3. **Best Practice**:
   - Develop and test locally in VS Code
   - Run final processing in Colab
   - Keep both `.py` and `.ipynb` versions in sync

---

## Quick Setup Commands

```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter

# Install jupytext for sync
pip install jupytext

# Sync py and ipynb
jupytext --set-formats ipynb,py ex3/rag_system.py
```

