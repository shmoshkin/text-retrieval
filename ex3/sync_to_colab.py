#!/usr/bin/env python3
"""
Quick script to sync notebook to Google Drive for Colab usage.
Run this after making changes in VS Code.
"""

import shutil
from pathlib import Path
import os

def find_google_drive():
    """Find Google Drive folder on common locations."""
    home = Path.home()
    
    # Common Google Drive locations
    possible_locations = [
        home / "Google Drive",
        home / "GoogleDrive", 
        home / "Library/CloudStorage/GoogleDrive-*",  # macOS
        Path("/content/drive/MyDrive"),  # Colab
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    # Check for environment variable
    drive_path = os.getenv("GOOGLE_DRIVE_PATH")
    if drive_path and Path(drive_path).exists():
        return Path(drive_path)
    
    return None

def sync_notebook():
    """Copy notebook to Google Drive folder."""
    notebook = Path("ex3/rag_system.ipynb")
    
    if not notebook.exists():
        print(f"❌ Notebook not found: {notebook}")
        return False
    
    drive_folder = find_google_drive()
    
    if not drive_folder:
        print("❌ Google Drive folder not found")
        print("\nOptions:")
        print("1. Set GOOGLE_DRIVE_PATH environment variable")
        print("2. Or manually copy the notebook to Drive")
        print(f"   Source: {notebook.absolute()}")
        return False
    
    # Create target folder if needed
    target_folder = drive_folder / "rag-assignment"
    target_folder.mkdir(exist_ok=True)
    
    # Copy notebook
    target = target_folder / "rag_system.ipynb"
    shutil.copy(notebook, target)
    
    print(f"✅ Synced notebook to: {target}")
    print(f"\n📝 Next steps:")
    print(f"1. Open Colab: https://colab.research.google.com/")
    print(f"2. File → Open Notebook → Google Drive")
    print(f"3. Navigate to: rag-assignment/rag_system.ipynb")
    
    return True

if __name__ == "__main__":
    sync_notebook()





