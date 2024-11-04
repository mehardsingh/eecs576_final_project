# Define the recall@k calculation function
def compute_recall_at_k(logits, labels, k=20):
    """
    Computes recall@k for a batch.
    
    Parameters:
        logits (torch.Tensor): The predicted logits for masked tokens, shape (num_masked_tokens, vocab_size).
        labels (torch.Tensor): The true labels for masked tokens, shape (num_masked_tokens,).
        k (int): The cutoff for recall (e.g., recall@20).
    
    Returns:
        float: The recall@k score for the batch.
    """
    top_k = logits.topk(k, dim=-1).indices  # Get top-k indices for each masked token
    correct = top_k.eq(labels.unsqueeze(1))  # Check if any of the top-k indices match the true label
    recall = correct.any(dim=1).float().mean().item()  # Calculate recall@k
    return recall

# Define the MRR calculation function
def compute_mrr(logits, labels):
    """
    Computes mean reciprocal rank (MRR) for a batch.
    
    Parameters:
        logits (torch.Tensor): The predicted logits for masked tokens, shape (num_masked_tokens, vocab_size).
        labels (torch.Tensor): The true labels for masked tokens, shape (num_masked_tokens,).
    
    Returns:
        float: The MRR score for the batch.
    """
    ranks = (logits.argsort(dim=-1, descending=True) == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    reciprocal_ranks = 1.0 / ranks.float()
    mrr = reciprocal_ranks.mean().item()  # Calculate mean reciprocal rank
    return mrr