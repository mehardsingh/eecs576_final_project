import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def mul_slope(self, alibi, slopes):
        # Assuming T is of shape BxSxS and slopes is of shape H
        B, S, S = alibi.shape
        H = slopes.shape[0]

        # Reshape slopes to broadcast over T
        # slopes becomes 1xHx1x1 to allow broadcasting
        alibi_expanded = alibi.unsqueeze(1)   # Bx1xSxS
        slopes_expanded = slopes.view(1, H, 1, 1)  # 1xHx1x1

        # Multiply with broadcasting, resulting in a BxHxSxS tensor
        alibi_final = alibi_expanded * slopes_expanded

        return alibi_final

    def calculate_slopes(self, num_heads):
        start = 2 ** (-8 / num_heads)
        ratio = start
        slopes = torch.tensor([start * (ratio ** i) for i in range(num_heads)])
        return slopes
        
    def forward(self, query, key, value, mask=None, dropout=None, alibi=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if alibi is not None:
            num_heads = query.shape[1]
            slopes = self.calculate_slopes(num_heads)
            alibi_final = self.mul_slope(alibi, slopes)
            scores = scores + alibi_final

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
