import pandas as pd
import numpy as np
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BartForConditionalGeneration, 
    BartTokenizer,
    PegasusForConditionalGeneration, 
    PegasusTokenizer
)
import nltk
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from tqdm import tqdm

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

class ResearchSummarizer:
    def __init__(self, model_name="google/pegasus-cnn_dailymail", device=None):
        """
        Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name (str): The name of the pre-trained model to use
            device (str): Device to use for computation (cpu or cuda)
        """
        self.model_name = model_name
        
        # Determine the device to use
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load the appropriate model and tokenizer based on model name
        try:
            if "t5" in model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
                self.prefix = "summarize: "
            elif "bart" in model_name.lower():
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
                self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
                self.prefix = ""
            elif "pegasus" in model_name.lower():
                # Add use_fast=False to avoid using the rust-based tokenizer which requires additional dependencies
                self.tokenizer = PegasusTokenizer.from_pretrained(model_name, use_fast=False)
                self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
                self.prefix = ""
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except ImportError as e:
            print(f"Error loading model: {e}")
            print("Please install the required dependencies: pip install sentencepiece protobuf")
            raise
        
        # Set up evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
    def load_compscholar_dataset(self, file_path):
        """
        Load the Brain Dead CompScholar Dataset
        
        Args:
            file_path (str): Path to the dataset CSV file
            
        Returns:
            pd.DataFrame: The loaded dataset
        """
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} entries")
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset for summarization task
        
        Args:
            df (pd.DataFrame): The dataset to preprocess
            
        Returns:
            tuple: Processed features and labels
        """
        # Use the 'Document' column as input and 'Summary' column as output
        X = df['Document'].values
        y = df['Summary'].values
        
        # Split the dataset into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_val, y_train, y_val
    
    def generate_summary(self, text, max_length=150, min_length=50):
        """
        Generate a summary for the given text
        
        Args:
            text (str): The text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: The generated summary
        """
        # Ensure the text is not too long for the model
        if len(text.split()) > 1024:
            # If text is too long, split into sentences and take first 1024 words
            sentences = sent_tokenize(text)
            truncated_text = ""
            word_count = 0
            
            for sentence in sentences:
                words = sentence.split()
                if word_count + len(words) <= 1024:
                    truncated_text += sentence + " "
                    word_count += len(words)
                else:
                    break
            
            text = truncated_text.strip()
        
        # Prepare the input for the model
        input_text = f"{self.prefix}{text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Generate the summary
        summary_ids = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
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
            generated_summary = self.generate_summary(text)
            
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
    
    def fine_tune(self, train_texts, train_summaries, validation_texts=None, validation_summaries=None, 
                  epochs=3, batch_size=4, learning_rate=5e-5, output_dir="fine_tuned_model"):
        """
        Fine-tune the model on the training data
        
        Args:
            train_texts (list): List of training texts
            train_summaries (list): List of training summaries
            validation_texts (list): List of validation texts
            validation_summaries (list): List of validation summaries
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for training
            output_dir (str): Directory to save the fine-tuned model
            
        Returns:
            dict: Training history
        """

        
        # Prepare the training dataset
        train_encodings = self.tokenizer([f"{self.prefix}{text}" for text in train_texts], 
                                         padding='max_length', truncation=True, max_length=1024)
        train_labels = self.tokenizer(train_summaries, 
                                     padding='max_length', truncation=True, max_length=128)
        
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings.input_ids,
            'attention_mask': train_encodings.attention_mask,
            'labels': train_labels.input_ids
        })
        
        # Prepare the validation dataset if provided
        if validation_texts and validation_summaries:
            val_encodings = self.tokenizer([f"{self.prefix}{text}" for text in validation_texts], 
                                          padding='max_length', truncation=True, max_length=1024)
            val_labels = self.tokenizer(validation_summaries, 
                                       padding='max_length', truncation=True, max_length=128)
            
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings.input_ids,
                'attention_mask': val_encodings.attention_mask,
                'labels': val_labels.input_ids
            })
        else:
            val_dataset = None
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            save_strategy="epoch",
            evaluation_strategy="epoch" if val_dataset else "no",
            weight_decay=0.01,
            fp16=True if self.device == "cuda" else False
        )
        
        # Initialize the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer.state.log_history
    
    def visualize_results(self, metrics, model_names):
        """
        Visualize the evaluation results
        
        Args:
            metrics (list): List of dictionaries containing metrics for each model
            model_names (list): List of model names corresponding to the metrics
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot ROUGE-1 scores
        rouge1_scores = [metric['rouge1'] for metric in metrics]
        axs[0, 0].bar(model_names, rouge1_scores)
        axs[0, 0].set_title('ROUGE-1 Scores')
        axs[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot ROUGE-2 scores
        rouge2_scores = [metric['rouge2'] for metric in metrics]
        axs[0, 1].bar(model_names, rouge2_scores)
        axs[0, 1].set_title('ROUGE-2 Scores')
        axs[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot ROUGE-L scores
        rougeL_scores = [metric['rougeL'] for metric in metrics]
        axs[1, 0].bar(model_names, rougeL_scores)
        axs[1, 0].set_title('ROUGE-L Scores')
        axs[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        
        # Plot BLEU scores
        bleu_scores = [metric['bleu'] for metric in metrics]
        axs[1, 1].bar(model_names, bleu_scores)
        axs[1, 1].set_title('BLEU Scores')
        axs[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('summarization_results.png')
        plt.show()

if __name__ == "__main__":
    # Initialize the summarizer
    summarizer = ResearchSummarizer()
    
    # Load the dataset
    df = summarizer.load_compscholar_dataset('/workspaces/brain-dead-2k25/Brain Dead CompScholar Dataset.csv')
    
    # Preprocess the data
    X_train, X_val, y_train, y_val = summarizer.preprocess_data(df)
    
    # Fine-tune the model
    summarizer.fine_tune(X_train, y_train, X_val, y_val, epochs=3, batch_size=2)
    
    # Evaluate the model
    metrics = summarizer.evaluate(X_val, y_val)
    print(f"Evaluation metrics: {metrics}")
