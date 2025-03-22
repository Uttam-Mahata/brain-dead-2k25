#!/usr/bin/env python
# Download and prepare arXiv dataset for research article summarization

import os
import sys
import json
import subprocess
from pathlib import Path
import kagglehub

def install_dependencies():
    """Ensure required dependencies are installed"""
    try:
        print("✅ kagglehub already installed")
    except ImportError:
        print("📦 Installing kagglehub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        print("✅ kagglehub installed successfully")

def download_arxiv_dataset():
    """Download arXiv dataset using kagglehub"""
    print("🔄 Downloading arXiv dataset...")
    
    # Download the dataset
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print(f"✅ Dataset downloaded to: {path}")
    return path

def process_dataset(download_path):
    """Process the downloaded dataset and move to current directory"""
    # Create data directory if it doesn't exist
    data_dir = Path('./data/arxiv')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # List all downloaded files
    download_path = Path(download_path)
    files = list(download_path.glob('**/*'))
    
    print(f"📁 Found {len(files)} files in the downloaded dataset")
    
    # Copy and process files to our data directory
    for file_path in files:
        if file_path.is_file():
            # Create target path in our data directory
            relative_path = file_path.relative_to(download_path)
            target_path = data_dir / relative_path
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            print(f"📄 Copying {relative_path} to {target_path}")
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                dst.write(src.read())

def sample_data(data_dir, num_samples=5):
    """Show sample of dataset contents"""
    json_files = list(Path(data_dir).glob('**/*.json'))
    if not json_files:
        json_files = list(Path(data_dir).glob('**/*.jsonl'))
    
    if not json_files:
        print("⚠️ No JSON files found to sample from")
        return
    
    print(f"\n📊 Showing sample of {min(num_samples, len(json_files))} records from the dataset:")
    
    for i, file_path in enumerate(json_files[:num_samples]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # Handle jsonl format (read first line)
                    data = json.loads(f.readline().strip())
                else:
                    # Handle json format
                    data = json.load(f)
                
                if isinstance(data, list) and data:
                    # If it's a list, show the first item
                    print(f"\nSample {i+1} from {file_path}:")
                    print(json.dumps(data[0], indent=2)[:500] + "...")
                else:
                    # Show the object
                    print(f"\nSample {i+1} from {file_path}:")
                    print(json.dumps(data, indent=2)[:500] + "...")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

def main():
    """Main function to download and process the arXiv dataset"""
    print("🚀 Starting arXiv dataset download and processing")
    
    # Install dependencies
    install_dependencies()
    
    # Download dataset
    download_path = download_arxiv_dataset()
    
    # Process dataset
    process_dataset(download_path)
    
    # Sample data
    sample_data('./data/arxiv')
    
    print("\n✅ arXiv dataset downloaded and processed successfully!")
    print("📂 Dataset is ready for use in your research article summarization task")
    print("\nTo use the dataset in your code:")
    print("```python")
    print("import json")
    print("from pathlib import Path")
    print("")
    print("# Load arXiv dataset")
    print("arxiv_files = list(Path('./data/arxiv').glob('**/*.json'))")
    print("# Process each file")
    print("for file_path in arxiv_files:")
    print("    with open(file_path, 'r') as f:")
    print("        data = json.load(f)")
    print("        # Process your data here")
    print("```")

if __name__ == "__main__":
    main()