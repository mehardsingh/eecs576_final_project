import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, pe_type="sinusoidal", max_len=512):
        super().__init__()

        # sinusoidal pe
        if pe_type == "sinusoidal":
            # Compute the positional encodings once in log space.
            pe = torch.zeros(max_len, d_model).float()
            pe.require_grad = False

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)

        # learned pe
        elif pe_type == "lpe":
            pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)
            pe = pe.unsqueeze(0)
            nn.init.xavier_uniform_(pe)

        # no pe
        else:
            pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)
            pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

