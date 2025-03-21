from summarizer import ResearchSummarizer
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compare_models(models, dataset_path, num_samples=None):
    """
    Compare different summarization models on the same dataset
    
    Args:
        models (list): List of model names to compare
        dataset_path (str): Path to the dataset CSV file
        num_samples (int, optional): Number of samples to use for evaluation
        
    Returns:
        tuple: Model names and their corresponding metrics
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Use a subset of the data if specified
    if num_samples and num_samples < len(df):
        df = df.sample(num_samples, random_state=42)
    
    # Prepare the data
    documents = df['Document'].values
    reference_summaries = df['Summary'].values
    
    all_metrics = []
    successful_models = []
    
    # Evaluate each model
    for model_name in models:
        print(f"Evaluating model: {model_name}")
        try:
            # Initialize the summarizer with the current model
            summarizer = ResearchSummarizer(model_name=model_name)
            
            # Evaluate the model
            metrics = summarizer.evaluate(documents, reference_summaries)
            all_metrics.append(metrics)
            successful_models.append(model_name)
            
            print(f"Metrics for {model_name}: {metrics}")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            print(f"Skipping model {model_name}")
            continue
    
    return successful_models, all_metrics

def visualize_comparison(model_names, metrics):
    """
    Visualize the comparison of different models
    
    Args:
        model_names (list): List of model names
        metrics (list): List of dictionaries containing metrics for each model
    """
    # Create a dataframe for visualization
    data = []
    for name, metric in zip(model_names, metrics):
        for metric_name, value in metric.items():
            data.append({
                'Model': name,
                'Metric': metric_name,
                'Score': value
            })
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df)
    plt.title('Comparison of Summarization Models')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Create a table to display the results
    pivot_df = df.pivot(index='Model', columns='Metric', values='Score')
    print("\nModel Comparison Results:")
    print(pivot_df)
    
    # Save the results to a CSV file
    pivot_df.to_csv('model_comparison_results.csv')
    
    return pivot_df

if __name__ == "__main__":
    # List of models to compare
    models_to_compare = [
        'google/pegasus-cnn_dailymail',
        'facebook/bart-large-cnn',
        't5-small',
        'google/pegasus-pubmed',
        'facebook/bart-large-xsum'
    ]
    
    # Compare the models
    model_names, all_metrics = compare_models(
        models_to_compare, 
        '/workspaces/brain-dead-2k25/Brain Dead CompScholar Dataset.csv',
        num_samples=20  # Use a small subset for quick evaluation
    )
    
    # Visualize the results
    results_df = visualize_comparison(model_names, all_metrics)
    
    # Find the best model for each metric
    best_models = {}
    for metric in results_df.columns:
        best_model = results_df[metric].idxmax()
        best_score = results_df[metric].max()
        best_models[metric] = (best_model, best_score)
    
    print("\nBest Model for Each Metric:")
    for metric, (model, score) in best_models.items():
        print(f"{metric}: {model} (Score: {score:.4f})")
