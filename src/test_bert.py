from bert.embedding.position import PositionalEmbedding
import torch

x = torch.arange(5).unsqueeze(0)

pe_cls = PositionalEmbedding(4, "sinusoidal", max_len=5)
pe = pe_cls(x)
print(pe)

pe_cls = PositionalEmbedding(4, "lpe", max_len=5)
pe = pe_cls(x)
print(pe)

pe_cls = PositionalEmbedding(4, "npe", max_len=5)
pe = pe_cls(x)
print(pe)