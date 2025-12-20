# Using Colab File Explorer (No Drive Mounting)

## Quick Setup

### Step 1: Upload Files to Colab

1. **Open Colab**: https://colab.research.google.com/
2. **Upload Notebook**: File → Upload Notebook → Select `rag_system.ipynb`
3. **Upload Data Files**:
   - Click **📁 folder icon** on the left sidebar
   - Click **"New Folder"** → Name it `data`
   - Click **"Upload"** button
   - Upload `train.csv` and `test.csv` to the `data` folder

### Step 2: Run the Notebook

1. **Run Cell 0** (Colab Setup) - installs packages
   - **No Drive mounting needed!** The setup cell skips Drive mounting
2. **Run all cells** (Runtime → Run All)

That's it! 🎉

---

## File Structure in Colab

After uploading, your Colab environment should look like:

```
/content/
├── data/
│   ├── train.csv
│   └── test.csv
├── rag_system.ipynb
└── predictions.csv (generated after running)
```

---

## What Changed

The notebook now:
- ✅ **Skips Google Drive mounting** if you're using file explorer
- ✅ **Automatically detects** files in Colab file explorer
- ✅ **Uses correct paths** (`data/` folder instead of `ex3/data`)
- ✅ **Shows warnings** if files are missing

---

## Troubleshooting

### "CSV files not found" warning

**Solution**: 
1. Check that files are uploaded to `/content/data/` folder
2. Verify file names: `train.csv` and `test.csv` (case-sensitive)
3. Refresh file browser (click folder icon again)

### Files uploaded but still not found

**Solution**:
- Make sure files are in a `data` folder (not `ex3/data`)
- Or update `DATA_DIR` in the config cell to match your folder structure

### Want to use Google Drive instead?

If you prefer Google Drive:
1. Uncomment the Drive mounting code in Cell 0
2. Update `DATA_DIR` path in config cell to point to Drive location

---

## Benefits of File Explorer (vs Drive)

✅ **Faster**: No need to mount Drive  
✅ **Simpler**: Just upload and run  
✅ **No authentication**: No Drive permissions needed  
✅ **Works offline**: Files are in Colab session  

**Note**: Files uploaded to Colab file explorer are **temporary** - they disappear when the session ends. For permanent storage, use Google Drive.




