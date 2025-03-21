#!/bin/bash

# Install required Python packages
pip install -r requirements.txt

# Make sure SentencePiece is properly installed
pip install sentencepiece protobuf

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "All dependencies installed successfully"
