import json
from collections import Counter
import torch
import argparse
from metrics import compute_recall_at_k, compute_mrr

def load_product2token(filepath):
    """
    Loads the product2token mapping from a JSON file.
    
    Args:
        filepath (str): File path to the product2token mapping.
        
    Returns:
        dict: A dictionary mapping product IDs to zero-indexed values.
    """
    with open(filepath, 'r') as file:
        product2token = json.load(file)
    return product2token

def get_product_popularity(train_fp):
    """
    Analyzes the training dataset to count product occurrences.
    
    Args:
        train_fp (str): File path to the training dataset.
        
    Returns:
        product_counts (Counter): Counter object with product IDs as keys and their counts as values.
    """
    product_counts = Counter()
    
    # Read and count product occurrences
    with open(train_fp, 'r') as file:
        for line in file:
            user_data = json.loads(line)
            product_interactions = user_data["products"]
            product_counts.update(product_interactions)
    
    return product_counts

def create_baseline_recommendation(train_fp, k=20):
    """
    Generates a baseline recommendation based on product popularity.
    
    Args:
        train_fp (str): File path to the training dataset.
        k (int): Number of top popular products to recommend.
        
    Returns:
        popular_products (list): List of top-k popular product IDs.
    """
    product_counts = get_product_popularity(train_fp)
    
    # Get top-k products by popularity
    popular_products = [product for product, count in product_counts.most_common(k)]
    
    return popular_products

def create_logits(product2token, product_counts):
    """
    Create a logits tensor with scores relative to product counts in the training data.
    
    Args:
        product2token (dict): Mapping of product IDs to zero-indexed values.
        product_counts (Counter): Counter object with product IDs as keys and their counts as values.
        
    Returns:
        logits (torch.Tensor): Tensor of shape (1, num_products) with scores for each product.
    """
    num_products = max(product2token.values()) + 1
    logits = torch.zeros(num_products)
    total_count = sum(product_counts.values())

    for product, count in product_counts.items():
        if product in product2token:
            idx = product2token[product]
            logits[idx] = count / total_count  # Normalize the count to get the score

    return logits.unsqueeze(0)  # Shape (1, num_products)

def calculate_metrics(val_fp, product2token, product_counts, k=20):
    """
    Calculates recall@k and MRR metrics for the baseline recommendation.
    
    Args:
        val_fp (str): File path to the validation dataset.
        product2token (dict): Mapping of product IDs to zero-indexed values.
        product_counts (Counter): Counter object with product IDs as keys and their counts as values.
        k (int): The cutoff for recall and MRR calculations.
        
    Returns:
        metrics (dict): Dictionary containing recall@k and MRR scores.
    """
    recall_scores = []
    mrr_scores = []
    
    logits = create_logits(product2token, product_counts)
    
    with open(val_fp, 'r') as file:
        for line in file:
            user_data = json.loads(line)
            true_products = user_data["products"]
            true_product = true_products[-1]  # Only consider the last product in the sequence
            
            if str(true_product) in product2token:
                true_product_idx = product2token[str(true_product)]
                recall_at_k = compute_recall_at_k(logits, torch.tensor([true_product_idx]), k=k)
                mrr = compute_mrr(logits, torch.tensor([true_product_idx]))
                
                recall_scores.append(recall_at_k)
                mrr_scores.append(mrr)
    
    average_recall_at_k = sum(recall_scores) / len(recall_scores)
    average_mrr = sum(mrr_scores) / len(mrr_scores)
    
    metrics = {
        "recall@k": average_recall_at_k,
        "mrr": average_mrr
    }
    
    return metrics

def main(args):
    product2token = load_product2token(args.product2token_fp)
    product_counts = get_product_popularity(args.train_ds_fp)
    metrics = calculate_metrics(args.val_ds_fp, product2token, product_counts, k=args.top_k)
    print(f"Recall@{args.top_k}: {metrics['recall@k']}")
    print(f"MRR: {metrics['mrr']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popularity-based baseline recommender.")
    parser.add_argument("--product2token_fp", type=str, default="data/product2token.json", help="File path to the product2token mapping.")
    parser.add_argument("--train_ds_fp", type=str, default="data/splits/train.jsonl", help="File path to the training dataset.")
    parser.add_argument("--val_ds_fp", type=str, default="data/splits/val.jsonl", help="File path to the validation dataset.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top popular products to recommend.")
    args = parser.parse_args()
    
    main(args)