import torch
import torch.nn as nn
from models.bert.bert import BERT

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x, alibi=None):
        h = self.bert(x, alibi)
        h = self.cls_head(h)
        
        E = self.bert.embedding.token.weight
        E = torch.permute(E, (1, 0))
        logits = torch.matmul(h, E)

        return logits
      
      
      