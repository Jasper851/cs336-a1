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



def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = (d_model + 12) // 24 * 64
        self.d_ff = d_ff
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        return self.w2(silu(w1x)*self.w3(x))



class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        dimension_indices = torch.arange(0,d_k,2,device=device,dtype=torch.float32)
        inverse_freq = theta ** (-dimension_indices / d_k)
        positions = torch.arange(max_seq_len,device=device,dtype=torch.float32)
        angles = torch.outer(positions,inverse_freq)
        self.register_buffer("cos_cached",torch.cos(angles),persistent=False)
        self.register_buffer("sin_cached",torch.sin(angles),persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)
        x_pairs = x.reshape(*x.shape[:-1],self.d_k//2,2)
        x_even = x_pairs[...,0]
        x_odd = x_pairs[...,1]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        rotated_pairs = torch.stack((rotated_even,rotated_odd),dim=-1)
        return rotated_pairs.reshape_as(x)



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_to_subtract = torch.max(x,dim=dim,keepdim=True).values
    tricked_x = x - max_to_subtract
    exponentials = torch.exp(tricked_x)
    denominator = torch.sum(exponentials,dim=dim,keepdim=True)
    return exponentials / denominator



def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    K_T = rearrange(K, "... keys d_k -> ... d_k keys")
    d_k = Q.shape[-1]
    similarity = einsum(Q, K_T, "... queries d_k, ... d_k keys -> ... queries keys")
    shifted = similarity / d_k ** 0.5
    if mask is None:
        masked = shifted
    else:
        masked = torch.masked_fill(shifted,~mask,-torch.inf)
    softmaxed = softmax(masked,dim=-1)
    return einsum(softmaxed, V, "... queries keys, ... keys d_v -> ... queries d_v")



class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise RuntimeError("d_model cannot be divided by num_heads")
        
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model,d_model,device,dtype)
        self.k_proj = Linear(d_model,d_model,device,dtype)
        self.v_proj = Linear(d_model,d_model,device,dtype)
        self.output_proj = Linear(d_model,d_model,device,dtype)
        self.with_rope = False
        if theta is not None:
            self.with_rope = True
            self.rope = RoPE(theta,self.d_k,max_seq_len,device)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads = self.num_heads)
        K = rearrange(K, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads = self.num_heads)
        V = rearrange(V, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads = self.num_heads)
        sequence_length = x.shape[-2]
        if self.with_rope:
            if token_positions is None:
                token_positions = torch.arange(end=sequence_length,device=x.device,dtype=torch.long)
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        mask = torch.ones((sequence_length,sequence_length),device=x.device,dtype=torch.bool)
        mask = torch.triu(mask,diagonal=1)
        mask = ~mask
        attention = scaled_dot_product_attention(Q,K,V,mask)
        attention = rearrange(attention, "... num_heads sequence_length d_k -> ... sequence_length (num_heads d_k)")
        return self.output_proj(attention)



class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.attn = Multihead_self_attention(d_model,num_heads,theta,max_seq_len,device,dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn = SwiGLU(d_model,d_ff,device,dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x),token_positions)
        output = y + self.ffn(self.ln2(y))
        return output



class Transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device = None, dtype = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model,device,dtype)
        self.layers = nn.ModuleList([Transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,device,dtype) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model,device=device,dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size,device,dtype)


    def forward(self, indices: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x,token_positions)
        return self.lm_head(self.ln_final(x))