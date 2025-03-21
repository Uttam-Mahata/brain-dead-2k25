import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import argparse
from datetime import datetime

from summarizer import ResearchSummarizer
from hybrid_summarizer import HybridSummarizer
from model_comparison import compare_models, visualize_comparison

def run_experiment(dataset_path, experiment_name, models=None, hybrid_methods=None, num_samples=None):
    """
    Run a summarization experiment comparing different models and approaches
    
    Args:
        dataset_path (str): Path to the dataset CSV file
        experiment_name (str): Name of the experiment
        models (list): List of model names to compare
        hybrid_methods (list): List of hybrid methods to compare
        num_samples (int, optional): Number of samples to use for evaluation
        
    Returns:
        tuple: Results of the experiment
    """
    # Create a directory for the experiment results
    output_dir = f"experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Use a subset of the data if specified
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
    
    # Prepare the data
    documents = df['Document'].values
    reference_summaries = df['Summary'].values
    
    results = {}
    
    # Evaluate standard models if specified
    if models:
        print("Evaluating standard models...")
        try:
            model_names, model_metrics = compare_models(models, dataset_path, num_samples)
            
            for name, metrics in zip(model_names, model_metrics):
                results[name] = metrics
            
            # Save the standard model results
            with open(f"{output_dir}/standard_models_results.json", "w") as f:
                json.dump({name: metrics for name, metrics in zip(model_names, model_metrics)}, f, indent=4)
        except ImportError as e:
            print(f"Error during model evaluation: {e}")
            print("Installing required dependencies and continuing with hybrid methods...")
            # If SentencePiece is missing, try to install it
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "protobuf"])
                print("Successfully installed missing dependencies")
            except:
                print("Failed to automatically install dependencies. Please install them manually.")
    
    # Evaluate hybrid approaches if specified
    if hybrid_methods:
        print("Evaluating hybrid approaches...")
        hybrid_results = {}
        
        for method in hybrid_methods:
            print(f"Evaluating hybrid method: {method}")
            
            # Initialize the hybrid summarizer
            summarizer = HybridSummarizer(extractive_model="bert-base-uncased", 
                                          abstractive_model="google/pegasus-cnn_dailymail")
            
            method_name, extractive_method, extractive_ratio = method
            
            # Evaluate on a sample of the data
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            bleu_scores = []
            
            for doc, ref in tqdm(zip(documents, reference_summaries), total=len(documents)):
                # Generate a summary using the hybrid approach
                summary = summarizer.hybrid_summarize(
                    doc, 
                    extractive_method=extractive_method,
                    extractive_ratio=extractive_ratio
                )
                
                # Calculate ROUGE scores
                rouge_scores = summarizer.rouge_scorer.score(ref, summary)
                rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
                rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
                rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
                
                # Calculate BLEU score
                bleu_score = summarizer.bleu.corpus_score([summary], [[ref]]).score
                bleu_scores.append(bleu_score)
            
            # Calculate average scores
            metrics = {
                'rouge1': np.mean(rouge1_scores),
                'rouge2': np.mean(rouge2_scores),
                'rougeL': np.mean(rougeL_scores),
                'bleu': np.mean(bleu_scores)
            }
            
            hybrid_results[method_name] = metrics
            results[method_name] = metrics
            
        # Save the hybrid results
        with open(f"{output_dir}/hybrid_methods_results.json", "w") as f:
            json.dump(hybrid_results, f, indent=4)
    
    # Create a summary visualization
    plt.figure(figsize=(14, 10))
    
    # Set up the data for visualization
    model_names = list(results.keys())
    rouge1_scores = [results[name]['rouge1'] for name in model_names]
    rouge2_scores = [results[name]['rouge2'] for name in model_names]
    rougeL_scores = [results[name]['rougeL'] for name in model_names]
    bleu_scores = [results[name]['bleu'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.2
    
    # Create the grouped bar chart
    plt.bar(x - 1.5*width, rouge1_scores, width, label='ROUGE-1', color='skyblue')
    plt.bar(x - 0.5*width, rouge2_scores, width, label='ROUGE-2', color='royalblue')
    plt.bar(x + 0.5*width, rougeL_scores, width, label='ROUGE-L', color='darkblue')
    plt.bar(x + 1.5*width, bleu_scores, width, label='BLEU', color='lightgreen')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(f'Summarization Results: {experiment_name}')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/results_visualization.png")
    
    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'Model': model_names,
        'ROUGE-1': rouge1_scores,
        'ROUGE-2': rouge2_scores,
        'ROUGE-L': rougeL_scores,
        'BLEU': bleu_scores
    })
    
    # Save the results to a CSV file
    results_df.to_csv(f"{output_dir}/results.csv", index=False)
    
    # Find the best model for each metric
    best_models = {}
    for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.iloc[best_idx]['Model']
        best_score = results_df.iloc[best_idx][metric]
        best_models[metric] = (best_model, best_score)
    
    # Save the best models to a file
    with open(f"{output_dir}/best_models.json", "w") as f:
        json.dump({metric: {"model": model, "score": float(score)} for metric, (model, score) in best_models.items()}, f, indent=4)
    
    print("\nBest Model for Each Metric:")
    for metric, (model, score) in best_models.items():
        print(f"{metric}: {model} (Score: {score:.4f})")
    
    return results, results_df, best_models

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description='Run summarization experiments')
    parser.add_argument('--dataset', type=str, default='/workspaces/brain-dead-2k25/Brain Dead CompScholar Dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--name', type=str, default=f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Name of the experiment')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples to use for evaluation')
    args = parser.parse_args()
    
    # Check for dependencies before running
    try:
        from transformers import PegasusTokenizer
        # Test if SentencePiece is properly installed
        _ = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail", use_fast=False)
        print("All dependencies are properly installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required dependencies...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "protobuf"])
            print("Successfully installed missing dependencies")
        except:
            print("Failed to automatically install dependencies. Please install them manually using:")
            print("pip install sentencepiece protobuf")
            if input("Continue without these dependencies? (y/n): ").lower() != 'y':
                sys.exit(1)
    
    # Define models to compare
    models_to_compare = [
        'google/pegasus-cnn_dailymail',
        'facebook/bart-large-cnn',
        't5-small',
        'google/pegasus-pubmed',
        'facebook/bart-large-xsum'
    ]
    
    # Define hybrid methods to compare
    # Format: (method_name, extractive_method, extractive_ratio)
    hybrid_methods = [
        ('Hybrid_TextRank_0.3', 'textrank', 0.3),
        ('Hybrid_TextRank_0.5', 'textrank', 0.5),
        ('Hybrid_TextRank_0.7', 'textrank', 0.7),
        ('Hybrid_KMeans_0.5', 'kmeans', 0.5),
        ('Hybrid_Embeddings_0.5', 'embeddings', 0.5)
    ]
    
    # Run the experiment
    results, results_df, best_models = run_experiment(
        dataset_path=args.dataset,
        experiment_name=args.name,
        models=models_to_compare,
        hybrid_methods=hybrid_methods,
        num_samples=args.samples
    )
    
    print(f"\nExperiment '{args.name}' completed successfully.")
    print(f"Results saved to 'experiments/{args.name}'")
