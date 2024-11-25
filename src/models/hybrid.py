import torch
import torch.nn as nn
from models.bert.bert import BERT

import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, pe_type="lpe"):
        super(BERT4Rec, self).__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout, pe_type)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x, alibi=None):
        h = self.bert(x, alibi)
        h = self.cls_head(h) # [B, S, D]

        return h[:, -1, :] # [B, D]

class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        
        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        return s_h

class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, num_layers):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=num_layers)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        embedding = self.embedding(x).squeeze()
        hidden = self.gated(embedding, edge_index)
        hidden2 = F.relu(hidden)
  
        return hidden2
    
class SRGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, num_layers):
        super(SRGNN, self).__init__()
        self.hidden_size = hidden_size
        self.gnn = GNNModel(hidden_size, n_node, num_layers)
        self.e2s = Embedding2Score(hidden_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.gnn(data)
        return self.e2s(h, self.gnn.embedding, batch)
      
class Hybrid(nn.Module):
    def __init__(self, bert4rec, srgnn, hidden_dim):
        super(Hybrid, self).__init__()
        
        self.bert4rec = bert4rec
        self.srgnn = srgnn

        self.bert_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.srgnn_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, seq, graph, alibi=None):
        h_bert = self.bert_proj(self.bert4rec(seq, alibi)) # [B, D]
        h_srgnn = self.srgnn_proj(self.srgnn(graph)) # [B, D]

        h = torch.cat((h_bert, h_srgnn), dim=-1) # [B, 2D]
        h = self.embedding_head(h) # [B, D]

        E = self.bert4rec.bert.embedding.token.weight
        E = torch.permute(E, (1, 0)) # [D, V]
        logits = torch.matmul(h, E) # [B, V]

        return logits



'''
bert = BERT()
bert_embedding = copy.deepcopy(bert.emebdding)

srgnn = SRGNN()
srgg.gnn.embedding = bert_embedding

hybrid = Hyrbid(bert, srgnn, hidden_size)
freeze hybrid.bert
'''
