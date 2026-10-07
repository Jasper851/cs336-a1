import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, device = None, dtype = None):
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass




class embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        pass


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass