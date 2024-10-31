import torch
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ECommerceDS(Dataset):
    def __init__(self, filepath, max_len, product2token, padding_token=-1, mask_token=-1, repeat_edges=True, mask=0.0):
        self.filepath = filepath
        self.max_len = max_len
        self.product2token = product2token
        self.padding_token = padding_token
        self.mask_token = mask_token
        self.repeat_edges = repeat_edges
        self.mask = mask
        self.file_reader = FileReader(file_path=filepath)

    def __len__(self):
        return len(self.file_reader)

    def left_pad(self, tensor, target_size, pad_value=0):
        """Left pad a tensor until it reaches the target size, return a binary mask and padding count."""
        padding_needed = target_size - tensor.size(0)  # Assuming 1D tensor
        padding_count = 0  # Initialize padding count

        if padding_needed > 0:
            # Create a tensor of padding values
            padding_tensor = torch.full((padding_needed,), pad_value, dtype=tensor.dtype)
            # Concatenate the padding tensor with the original tensor
            padded_tensor = torch.cat((padding_tensor, tensor))
            
            # Create a binary mask: 0 for padding, 1 for original values
            mask = torch.cat((torch.zeros(padding_needed, dtype=torch.long), torch.ones(tensor.size(0), dtype=torch.long)))
            
            # Set padding count
            padding_count = padding_needed
        else:
            padded_tensor = tensor  # No padding needed
            mask = torch.ones(tensor.size(0), dtype=torch.long)  # All valid values

        return padded_tensor, mask, padding_count

    def create_cloze_mask(self, indices_tensor, p):
        """Select a subset of indices with probability p, ensuring at least one index is selected, 
        returning a binary mask where 1 means the index was selected."""
        
        # Generate random numbers and create a boolean mask for selection
        mask = torch.rand(len(indices_tensor)) < p
        
        # Ensure at least one index is selected
        if not mask.any():  # Check if no indices are selected
            random_index = torch.randint(len(indices_tensor), (1,)).item()  # Select a random index
            mask[random_index] = 1  # Force select this index

        # Convert boolean mask to binary mask (0s and 1s)
        binary_mask = mask.to(torch.long)  # Change to int type (0 and 1)

        return binary_mask
    
    def get_token_ids(self, products):
        """Map product IDs in the tensor to their corresponding tokens."""
        # Convert the products tensor to string format
        product_ids_as_strings = products.to(torch.int64).tolist()  # Ensure it's in the right dtype
        product_ids_as_strings = list(map(str, product_ids_as_strings))
        
        # Use a list comprehension to map each product ID to its corresponding token
        mapped_products = torch.tensor([self.product2token[pid] for pid in product_ids_as_strings], dtype=torch.long)
        
        return mapped_products
    
    def zero_index_products(self, unpadded_products, edge_type="outgoing"):
        # Dictionary to map each unique value to an index
        value_to_index = {}
        reindexed_tensor = []

        # Loop through the tensor and assign new indices
        for value in unpadded_products:
            if value.item() not in value_to_index:
                value_to_index[value.item()] = len(value_to_index)  # Assign the next available index
            reindexed_tensor.append(value_to_index[value.item()])

        # Convert the result to a tensor
        reindexed_tensor = torch.tensor(reindexed_tensor)
        return reindexed_tensor
    
    def compute_edges(self, zero_index_products, edge_type="outgoing"):
        if edge_type == "outgoing":
            src = zero_index_products[:-1]
            dst = zero_index_products[1:]
        elif edge_type == "incoming":
            src = zero_index_products[1:]
            dst = zero_index_products[:-1]

        edges = torch.stack((src, dst), dim=0)
        return edges
        
    def create_graph(self, unpadded_products):
        zero_index_products = self.zero_index_products(unpadded_products)

        all_edges = self.compute_edges(zero_index_products, edge_type="outgoing")
        
        unique_edges, freq = torch.unique(all_edges, dim=1, return_counts=True)
        source_counts = torch.bincount(all_edges[0])
        weights = torch.zeros(unique_edges.shape[1])
        # Calculate weights for each unique edge
        for i in range(unique_edges.shape[1]):
            source_node = unique_edges[0, i]
            weights[i] = freq[i] / source_counts[source_node]
        print("Unique edges:\n", unique_edges)
        print("Frequencies:\n", freq)
        print("Weights:\n", weights)

        return unique_edges

    def __getitem__(self, idx):
        line_str = self.file_reader.read_line_by_number(idx)
        sample = json.loads(line_str)

        # get products
        products = torch.tensor(sample["products"])
        if type(self.mask) == float:
            indices = torch.arange(products.shape[0])
            cloze_mask = self.create_cloze_mask(indices, self.mask)
        elif self.mask == "last":
            cloze_mask = torch.zeros_like(products)
            cloze_mask[-1] = 1
        
        masked_products = products.clone()
        masked_products[cloze_mask==1] = self.mask_token
        
        masked_products, attention_mask, _ = self.left_pad(masked_products, self.max_len, pad_value=self.padding_token)
        unpadded_products = products.clone()
        products, _, _ = self.left_pad(products, self.max_len, pad_value=self.padding_token)
        cloze_mask, _, _ = self.left_pad(cloze_mask, self.max_len, pad_value=0)

        masked_products = self.get_token_ids(masked_products)
        products = self.get_token_ids(products)
        unpadded_products = self.get_token_ids(unpadded_products)

        print(unpadded_products)
        edge_indices = self.create_graph(unpadded_products)
        print(edge_indices)

        return {
            "masked_products": masked_products, 
            "products": products, 
            "attention_mask": attention_mask,
            "cloze_mask": cloze_mask
        }

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.line_offsets = self.build_line_index()

    def build_line_index(self):
        """Builds an index of line offsets in the file for fast access."""
        offsets = []
        with open(self.file_path, 'r') as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def read_line_by_number(self, line_num: int) -> str:
        """
        Params:
            line_num: the sample index from file_path

        Returns:
            the string representing the sample dictionary
        """
        if line_num >= len(self.line_offsets):
            raise ValueError(f"Line number {line_num} is out of range.")

        with open(self.file_path, 'r') as f:
            # Seek to the correct position in the file
            f.seek(self.line_offsets[line_num])
            return f.readline().strip()
        
filepath = "data/splits/train.jsonl"
max_len = 50
product2token_fp = "data/product2token.json"
with open(product2token_fp, mode="r") as f:
    product2token = json.load(f)
repeat_edges = True
ds = ECommerceDS(filepath, max_len, product2token, padding_token=-2, mask_token=-1, repeat_edges=True, mask=.7)

print(ds[10000])