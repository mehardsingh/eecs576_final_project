from bert.embedding.position import PositionalEmbedding
from bert.bert import BERT
from bert4rec import BERT4Rec
import torch
from dataset import ECommerceDS
import json
from sr_gnn import GNNModel, SRGNN
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

dl = DataLoader(ds, batch_size=1, shuffle=True)

batch = next(iter(dl))

x = batch["masked_products"]
graph = batch["graph"]

bert = BERT4Rec(vocab_size=20000, hidden=128, n_layers=5, attn_heads=2)
bert_logits = bert(x)
print("bert logits", bert_logits.shape)

srgnn = SRGNN(hidden_size=128, n_node=20000)
srgnn_logits = srgnn(graph)
print("srgnn_logits", srgnn_logits.shape)