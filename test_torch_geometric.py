import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

# Define a GNN model with learnable embeddings for nodes 0-9
class GNNWithEmbeddings(torch.nn.Module):
    def __init__(self, num_total_nodes, embedding_dim, hidden_channels, out_channels):
        super(GNNWithEmbeddings, self).__init__()
        # Learnable node embeddings for all nodes (0-9)
        self.embeddings = torch.nn.Embedding(num_total_nodes, embedding_dim)
        
        # GNN layers (e.g., GCN)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        # Extract the nodes present in the current batch (e.g., nodes 1, 2, 3)
        node_ids = torch.unique(data.edge_index)  # Unique node IDs in the batch

        print("node_ids", torch.unique(data.x.flatten()))
        
        # Get embeddings for the nodes in the batch
        x = self.embeddings(node_ids)

        print("x", x.shape)
        
        # Pass embeddings through GNN layers
        x = self.conv1(x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        
        return x
    

edge_index = torch.tensor([[10, 0, 1, 1, 2],
                           [1, 2, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[0], [1], [2]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

data_list = [data]
loader = DataLoader(data_list, batch_size=1)

# Initialize the model
num_total_nodes = 100  # Embeddings for nodes 0 to 99
embedding_dim = 16  # Embedding size
hidden_channels = 32  # Hidden layer size for GNN
out_channels = 2  # Output size

model = GNNWithEmbeddings(num_total_nodes, embedding_dim, hidden_channels, out_channels)

# Example forward pass
for batch in loader:
    print(batch)
    pred = model(batch)
    break
