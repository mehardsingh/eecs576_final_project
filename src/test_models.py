# from bert.embedding.position import PositionalEmbedding
# from bert.bert import BERT
from models.bert4rec import BERT4Rec
import torch
from dataset import ECommerceDS
import json
from models.sr_gnn import GNNModel, SRGNN
from models.hybrid import BERT4Rec as BERT4Rec_H, SRGNN as SRGNN_H, Hybrid
from torch_geometric.loader import DataLoader

with open("data/product2token.json", mode="r") as f:
    product2token = json.load(f)

ds = ECommerceDS(
    filepath="data/splits/train.jsonl",
    max_len=50,
    product2token=product2token,
    padding_token=-2,
    mask_token=-1,
    mask=0.15
)

dl = DataLoader(ds, batch_size=1, shuffle=False)

batch = next(iter(dl))

print(batch.keys())
x = batch["masked_products"]
graph = batch["graph"]
alibi = batch["alibi"]

# bert = BERT4Rec(vocab_size=20000, hidden=128, n_layers=5, attn_heads=2)
# bert_logits = bert(x)
# print("bert logits", bert_logits.shape)

# srgnn = SRGNN(hidden_size=128, n_node=20000, num_layers=2)
# srgnn_logits = srgnn(graph)
# print("srgnn_logits", srgnn_logits.shape)

# OK
# print("products", batch["products"].shape)
# print(batch["products"])

# OK
# print("alibi", alibi.shape)
# print(alibi[0][40:50, 40:50])

# bert_tlab = BERT4Rec(vocab_size=20000, hidden=128, n_layers=5, attn_heads=2)
# bert_tlab_logits = bert_tlab(x, alibi=alibi)
# print("bert_tlab_logits", bert_tlab_logits.shape)

'''
bert = BERT()
bert_embedding = copy.deepcopy(bert.emebdding)

srgnn = SRGNN()
srgg.gnn.embedding = bert_embedding

hybrid = Hyrbid(bert, srgnn, hidden_size)
freeze hybrid.bert
'''

# parser.add_argument("--hidden_dim", type=int, default=128)
#     parser.add_argument("--n_layers", type=int, default=4)
#     parser.add_argument("--attn_heads", type=int, default=4)
#     parser.add_argument("--pe_type", type=str, default="lpe")

import copy

bert = BERT4Rec_H(vocab_size=22218, hidden=128, n_layers=4, attn_heads=4)
bert.load_state_dict(torch.load('results/bert_baseline/best_model.pth'))
E = copy.deepcopy(bert.bert.embedding.token.weight)

print(f">> Model params: {sum(p.numel() for p in bert.parameters() if p.requires_grad)/1e6:.3f}M")

srgnn = SRGNN_H(hidden_size=128, n_node=20000, num_layers=2)
hybrid = Hybrid(bert, srgnn, hidden_dim=128)

hybrid.srgnn.gnn.embedding.weight = E

# freeze params
hybrid.srgnn.gnn.embedding.weight.requires_grad = False
for param in hybrid.bert4rec.parameters():
    param.requires_grad = False

logits = hybrid(x, graph, alibi)

print("logits", logits.shape)
print("logits", logits)