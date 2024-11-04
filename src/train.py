from models.bert4rec import BERT4Rec
import torch
from dataset import ECommerceDS
from torch_geometric.loader import DataLoader
import json
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

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

with open("data/product2token.json", mode="r") as f:
    product2token = json.load(f)

train_ds = ECommerceDS(
    filepath="data/splits/train.jsonl",
    max_len=50,
    product2token=product2token,
    padding_token=-2,
    mask_token=-1,
    mask=0.15
)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
model = BERT4Rec(vocab_size=22218, hidden=128, n_layers=12, attn_heads=4)
print(f"Print # params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_recall = 0.0
    running_mrr = 0.0
    num_batches = 0

    for batch in tqdm(train_dl):
        x = batch["masked_products"]
        labels = batch["products"]
        cloze_mask = batch["cloze_mask"]

        optimizer.zero_grad()  # Zero the parameter gradients

        logits = model(x)  # Forward pass

        logits_flat = logits.view(-1, logits.shape[-1])  # Shape: (B*S, V)
        labels_flat = labels.view(-1)                    # Shape: (B*S,)
        cloze_mask_flat = cloze_mask.view(-1)            # Shape: (B*S,)

        # Select only the masked positions
        valid_indices = cloze_mask_flat == 1
        logits_masked = logits_flat[valid_indices]       # Shape: (num_masked_tokens, V)
        labels_masked = labels_flat[valid_indices]       # Shape: (num_masked_tokens,)

        # Calculate the loss only on masked tokens
        loss = F.cross_entropy(logits_masked, labels_masked)
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss

        # Calculate and accumulate recall@20
        recall_at_20 = compute_recall_at_k(logits_masked, labels_masked, k=20)
        running_recall += recall_at_20

        # Calculate and accumulate MRR
        mrr = compute_mrr(logits_masked, labels_masked)
        running_mrr += mrr

        num_batches += 1

        print(loss.item(), recall_at_20, mrr)

    # Print average loss, recall@20, and MRR for the epoch
    epoch_loss = running_loss / num_batches
    epoch_recall = running_recall / num_batches
    epoch_mrr = running_mrr / num_batches
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Recall@20: {epoch_recall:.4f}, MRR: {epoch_mrr:.4f}")
