import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        """
        in_features: int
        out_features: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        var = 2 / (in_features + out_features)
        std = var ** 0.5
        self.weight = torch.empty(out_features,in_features,device=device,dtype=dtype)
        self.weight = nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)
        self.weight = nn.Parameter(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return y




class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        """
        num_embeddings: int
        embedding_dim: int
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        self.weight = nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3)
        self.weight = nn.Parameter(self.weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]