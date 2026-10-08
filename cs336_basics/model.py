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



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        """
        d_model: int
        eps: float = 1e-5
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.ones(d_model,device=device,dtype=dtype)
        self.weight = nn.Parameter(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1,keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = x / rms
        return normalized.to(in_dtype) * self.weight



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        if not d_ff:
            d_ff = 64 * torch.floor(d_model / 24 + 0.5)
        self.d_ff = d_ff
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        return self.w2((w1x*torch.sigmoid(w1x))*self.w3(x))



