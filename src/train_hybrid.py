from models.hybrid import BERT4Rec, SRGNN, Hybrid
import torch
from dataset import ECommerceDS
from torch_geometric.loader import DataLoader
import json
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from metric_handler import MetricHandler
import random
from torch.utils.data import Subset
import os
import argparse
from metrics import compute_recall_at_k, compute_mrr
import copy

def load_model(product2token, args):
    # Initialize BERT model
    bert = BERT4Rec(
        vocab_size=len(product2token), 
        hidden=args.hidden_dim, 
        n_layers=args.n_layers_bert, 
        attn_heads=args.attn_heads, 
        pe_type=args.pe_type
    )
    bert.load_state_dict(torch.load(args.bert_pretrained_dir, map_location=args.device))
    bert.to(args.device)  # Ensure BERT is on the correct device
    
    # Extract and deepcopy embedding weights from BERT
    embedding_weights = bert.bert.embedding.token.weight.detach().to(args.device)

    # Initialize SRGNN model
    srgnn = SRGNN(
        hidden_size=args.hidden_dim, 
        n_node=len(product2token), 
        num_layers=args.n_layers_srgnn
    ).to(args.device)  # Move SRGNN to the correct device
    
    # Initialize Hybrid model combining BERT and SRGNN
    hybrid = Hybrid(bert, srgnn, hidden_dim=args.hidden_dim).to(args.device)

    # Assign BERT embedding weights to SRGNN and freeze them
    hybrid.srgnn.gnn.embedding.weight = nn.Parameter(embedding_weights)
    hybrid.srgnn.gnn.embedding.weight.requires_grad = False

    # Freeze all parameters of BERT
    for param in hybrid.bert4rec.parameters():
        param.requires_grad = False

    return hybrid

def main(args):
    with open(args.product2token_fp, mode="r") as f:
        product2token = json.load(f)

    train_ds = ECommerceDS(
        filepath=args.train_ds_fp,
        max_len=args.max_len,
        product2token=product2token,
        padding_token=-2,
        mask_token=-1,
        mask="last",
        graph_remove_last=True
    )
    val_ds = ECommerceDS(
        filepath=args.val_ds_fp,
        max_len=args.max_len,
        product2token=product2token,
        padding_token=-2,
        mask_token=-1,
        mask="last",
        graph_remove_last=True
    )

    if args.subsample:
        subset_indices = random.sample(range(len(train_ds)), args.subsample)
        subset_dataset = Subset(train_ds, subset_indices)
        train_dl = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True)

        subset_indices = random.sample(range(len(val_ds)), args.subsample)
        subset_dataset = Subset(val_ds, subset_indices)
        val_dl = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)

    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)        
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = load_model(product2token, args)
    print(f">> Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M")

    initial_lr = args.initial_lr
    final_lr = args.final_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.wd)

    metric_handler = MetricHandler(save_dir=args.save_dir)
    step_count = 0
    eval_steps = int(len(train_dl) * args.eval_steps)

    # Training loop
    total_iters = args.num_epochs * len(train_dl)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()  # Set model to training mode

        for i, batch in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            x = batch["masked_products"].to(args.device) # (B, S, V)
            graph = batch["graph"].to(args.device)
            alibi = batch["alibi"].to(args.device) # (B, S, S)
            labels = batch["products"].to(args.device) # (B, S)
            cloze_mask = batch["cloze_mask"].to(args.device) # (B, S)

            optimizer.zero_grad()  # Zero the parameter gradients

            logits_flat = model(x, graph, alibi=alibi)  # BxV
            labels_flat = labels[:,-1]                    # Shape: (B)
            cloze_mask_flat = cloze_mask[:,-1]                    # Shape: (B)

            # Select only the masked positions
            valid_indices = cloze_mask_flat == 1
            logits_masked = logits_flat[valid_indices]       # Shape: (B, V)
            labels_masked = labels_flat[valid_indices]       # Shape: (B,)

            # Calculate the loss only on masked tokens
            loss = F.cross_entropy(logits_masked, labels_masked)
            loss.backward()  # Backward pass

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()  # Update weights

            # Update the learning rate
            current_iter = epoch * len(train_dl) + i
            new_lr = initial_lr + (final_lr - initial_lr) * (current_iter / total_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            batch_metrics = {
                "train_loss": loss.item(),
                "train_recall@1": compute_recall_at_k(logits_masked, labels_masked, k=1),
                "train_recall@5": compute_recall_at_k(logits_masked, labels_masked, k=5),
                "train_recall@10": compute_recall_at_k(logits_masked, labels_masked, k=10),
                "train_recall@20": compute_recall_at_k(logits_masked, labels_masked, k=20),
                "train_mrr": compute_mrr(logits_masked, labels_masked)
            }
            metric_handler.batch_update(batch_metrics)
            step_count += 1

            # Evaluate model on validation set every N steps
            if step_count % eval_steps == 0:
                model.eval()  # Set model to evaluation mode

                with torch.no_grad():  # Disable gradient computation for evaluation
                    for val_batch in tqdm(val_dl, desc=f"Validating (step {step_count})"):
                        x_val = val_batch["masked_products"].to(args.device)
                        graph_val = val_batch["graph"].to(args.device)
                        alibi_val = val_batch["alibi"].to(args.device)
                        labels_val = val_batch["products"].to(args.device)
                        cloze_mask_val = val_batch["cloze_mask"].to(args.device)

                        logits_flat_val = model(x_val, graph_val, alibi=alibi_val)  # BxV
                        labels_flat_val = labels_val[:, -1] # Bx1
                        cloze_mask_flat_val = cloze_mask_val[:, -1] # Bx1

                        # Select only the masked positions
                        valid_indices_val = cloze_mask_flat_val == 1
                        logits_masked_val = logits_flat_val[valid_indices_val]
                        labels_masked_val = labels_flat_val[valid_indices_val]

                        # Calculate validation loss and metrics
                        val_loss = F.cross_entropy(logits_masked_val, labels_masked_val)
                        batch_metrics = {
                            "val_loss": val_loss.item(),
                            "val_recall@1": compute_recall_at_k(logits_masked_val, labels_masked_val, k=1),
                            "val_recall@5": compute_recall_at_k(logits_masked_val, labels_masked_val, k=5),
                            "val_recall@10": compute_recall_at_k(logits_masked_val, labels_masked_val, k=10),
                            "val_recall@20": compute_recall_at_k(logits_masked_val, labels_masked_val, k=20),
                            "val_mrr": compute_mrr(logits_masked_val, labels_masked_val),
                            "step_count": step_count
                        }
                        metric_handler.batch_update(batch_metrics)

                metric_handler.all_update_save_clear()
                model.train()

                # Check for early stopping
                if metric_handler.metric_dict["val_loss"][-1] < best_loss:
                    best_loss = metric_handler.metric_dict["val_loss"][-1]
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'curr_model.pth'))
                    print(f">> Step {step_count} | Best val loss, saving model")
                else:
                    patience_counter += 1
                    torch.save(model.state_dict(), os.path.join(args.save_dir, 'curr_model.pth'))

                if patience_counter >= args.patience:
                    print(">> Early stopping triggered")
                    break  # Stop training if the patience counter exceeds the threshold
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for argparse.")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--product2token_fp", type=str, default="data/product2token.json")
    parser.add_argument("--train_ds_fp", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val_ds_fp", type=str, default="data/splits/val.jsonl")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--eval_steps", type=float, default=1.0)
    parser.add_argument("--num_epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="results/hybrid/")

    parser.add_argument("--bert_pretrained_dir", type=str, default="results/bert_tlab_nope/best_model.pth")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers_bert", type=int, default=4)
    parser.add_argument("--n_layers_srgnn", type=int, default=2)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--pe_type", type=str, default="nope")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--initial_lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=1e-2)

    args = parser.parse_args()
    main(args)