import pandas as pd
import numpy as np
import torch
from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    BertModel,
    BertTokenizer,
    BartTokenizer,
    BartForConditionalGeneration
)
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class HybridSummarizer:
    def __init__(self, extractive_model="bert-base-uncased", abstractive_model="google/pegasus-cnn_dailymail", device=None):
        """
        Initialize the hybrid summarizer with extractive and abstractive models.
        
        Args:
            extractive_model (str): The name of the pre-trained model for extractive summarization
            abstractive_model (str): The name of the pre-trained model for abstractive summarization
            device (str): Device to use for computation (cpu or cuda)
        """
        self.extractive_model_name = extractive_model
        self.abstractive_model_name = abstractive_model
        
        # Determine the device to use
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load the extractive model
        try:
            self.bert_tokenizer = BertTokenizer.from_pretrained(extractive_model)
            self.bert_model = BertModel.from_pretrained(extractive_model).to(self.device)
            
            # Load the abstractive model
            if "pegasus" in abstractive_model.lower():
                self.pegasus_tokenizer = PegasusTokenizer.from_pretrained(abstractive_model, use_fast=False)
                self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(abstractive_model).to(self.device)
            else:
                print(f"Warning: Using {abstractive_model} instead of a Pegasus model")
                # Fallback to BART if Pegasus fails
                self.pegasus_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.pegasus_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        except ImportError as e:
            print(f"Error loading model: {e}")
            print("Falling back to BART model as a backup")
            # Fallback to BART if Pegasus fails due to missing sentencepiece
            self.bert_tokenizer = BertTokenizer.from_pretrained(extractive_model)
            self.bert_model = BertModel.from_pretrained(extractive_model).to(self.device)
            self.pegasus_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.pegasus_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        
        # Set up evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
    def extractive_summarization(self, text, num_sentences=5, method="textrank"):
        """
        Perform extractive summarization on the input text
        
        Args:
            text (str): The text to summarize
            num_sentences (int): Number of sentences to extract
            method (str): Method to use for extractive summarization ('textrank', 'kmeans', or 'embeddings')
            
        Returns:
            str: The extractive summary
        """
        # Split the text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        if method == "textrank":
            # TextRank algorithm
            # Create a similarity matrix
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create a graph
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences based on scores
            ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
            
            # Select top sentences
            selected_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
            
            # Preserve the original order of sentences
            ordered_selected_sentences = [s for s in sentences if s in selected_sentences]
            
            return " ".join(ordered_selected_sentences)
            
        elif method == "kmeans":
            # K-means clustering
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            
            # Apply K-means clustering
            num_clusters = min(num_sentences, len(sentences) - 1)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(tfidf_matrix)
            
            # Find the sentences closest to the centroids
            closest_sentences = []
            for cluster_idx in range(num_clusters):
                cluster_sentences = [i for i, label in enumerate(kmeans.labels_) if label == cluster_idx]
                
                if cluster_sentences:
                    # Find the sentence closest to the centroid
                    centroid = kmeans.cluster_centers_[cluster_idx]
                    distances = [np.linalg.norm(tfidf_matrix[i].toarray() - centroid) for i in cluster_sentences]
                    closest_idx = cluster_sentences[np.argmin(distances)]
                    closest_sentences.append(closest_idx)
            
            # Select the top sentences
            selected_sentences = [sentences[i] for i in sorted(closest_sentences)]
            
            return " ".join(selected_sentences)
            
        elif method == "embeddings":
            # BERT embeddings
            # Get BERT embeddings for each sentence
            embeddings = []
            for sentence in sentences:
                inputs = self.bert_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])
            
            # Compute cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create a graph
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph)
            
            # Rank sentences based on scores
            ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
            
            # Select top sentences
            selected_indices = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
            
            # Preserve the original order of sentences
            ordered_selected_indices = sorted(selected_indices)
            selected_sentences = [sentences[i] for i in ordered_selected_indices]
            
            return " ".join(selected_sentences)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def abstractive_summarization(self, text, max_length=150, min_length=50):
        """
        Perform abstractive summarization on the input text
        
        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: The abstractive summary
        """
        # Prepare the input for the model
        inputs = self.pegasus_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Generate the summary
        summary_ids = self.pegasus_model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Decode the summary
        summary = self.pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def hybrid_summarize(self, text, extractive_method="textrank", extractive_ratio=0.5, max_length=150, min_length=50):
        """
        Perform hybrid summarization by first extracting key sentences and then applying abstractive summarization
        
        Args:
            text (str): The text to summarize
            extractive_method (str): Method to use for extractive summarization
            extractive_ratio (float): Ratio of sentences to extract (0.0 to 1.0)
            max_length (int): Maximum length of the final summary
            min_length (int): Minimum length of the final summary
            
        Returns:
            str: The hybrid summary
        """
        # Calculate the number of sentences to extract
        sentences = sent_tokenize(text)
        num_sentences = max(1, int(len(sentences) * extractive_ratio))
        
        # Perform extractive summarization
        extractive_summary = self.extractive_summarization(text, num_sentences=num_sentences, method=extractive_method)
        
        # Perform abstractive summarization on the extractive summary
        hybrid_summary = self.abstractive_summarization(extractive_summary, max_length=max_length, min_length=min_length)
        
        return hybrid_summary
    
    def evaluate(self, texts, reference_summaries):
        """
        Evaluate the model performance on a set of texts
        
        Args:
            texts (list): List of input texts
            reference_summaries (list): List of reference summaries
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        for text, reference in tqdm(zip(texts, reference_summaries), total=len(texts), desc="Evaluating"):
            # Generate a summary
            generated_summary = self.hybrid_summarize(text)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, generated_summary)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate BLEU score
            bleu_score = self.bleu.corpus_score([generated_summary], [[reference]]).score
            bleu_scores.append(bleu_score)
        
        # Calculate average scores
        metrics = {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'bleu': np.mean(bleu_scores)
        }
        
        return metrics

if __name__ == "__main__":
    # Initialize the hybrid summarizer
    summarizer = HybridSummarizer()
    
    # Load the dataset
    df = pd.read_csv('/workspaces/brain-dead-2k25/Brain Dead CompScholar Dataset.csv')
    
    # Select a sample text for demonstration
    sample_text = df.iloc[61]['Document']
    reference_summary = df.iloc[61]['Summary']
    
    # Generate summaries using different methods
    print("Original text length:", len(sample_text.split()))
    
    extractive_summary = summarizer.extractive_summarization(sample_text, method="textrank")
    print("\nExtractive summary length:", len(extractive_summary.split()))
    print(extractive_summary)
    
    abstractive_summary = summarizer.abstractive_summarization(sample_text)
    print("\nAbstractive summary length:", len(abstractive_summary.split()))
    print(abstractive_summary)
    
    hybrid_summary = summarizer.hybrid_summarize(sample_text)
    print("\nHybrid summary length:", len(hybrid_summary.split()))
    print(hybrid_summary)
    
    print("\nReference summary length:", len(reference_summary.split()))
    print(reference_summary)
    
    # Evaluate on a small subset of the data
    sample_df = df.sample(10, random_state=42)
    sample_texts = sample_df['Document'].values
    sample_reference_summaries = sample_df['Summary'].values
    
    metrics = summarizer.evaluate(sample_texts, sample_reference_summaries)
    print("\nEvaluation metrics:", metrics)
