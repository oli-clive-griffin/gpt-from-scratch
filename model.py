import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_k,
        d_v,
        block_size,
    ):
        super().__init__()
        self.q = nn.Linear(d_in, d_k, bias=False)
        self.k = nn.Linear(d_in, d_k, bias=False)
        self.v = nn.Linear(d_in, d_v, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size)), diagonal=0))

    def forward(self, x: Tensor):
        B,T,C = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        xout = AttentionBlock._scaled_dot_product_attention(Q, K, V, self.tril[:T,:T])
        return xout

    @staticmethod
    def _scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor: # shape (batch_size, seq_length, dim_v)
        assert len(q.shape) == 3 and len(k.shape) == 3, 'expected both q and k to be 3d'
        attn_matrix = q @ k.transpose(-2, -1)
        if mask is not None:
            attn_matrix = attn_matrix.masked_fill(mask==0, float('-inf'))
        attn_matrix = attn_matrix.softmax(dim=-1)
        return (attn_matrix @ v) / (k.size(-1) ** 0.5)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        d_per_head,
        block_size,
    ):
        super().__init__()
        self.d_per_head = d_per_head
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionBlock(d_per_head, d_per_head, d_per_head, block_size) for _ in range(num_heads)])

    def forward(self, x: Tensor):
        d_in = x.shape[2]
        assert d_in % self.num_heads == 0
        assert d_in // self.num_heads == self.d_per_head
        d_per_head = d_in // self.num_heads

        xs = [
            head.forward(x[:, :, i * d_per_head : (i+1) * d_per_head])
            for i, head in enumerate(self.heads)
        ]

        return torch.cat(xs, dim=2)


class FeedForward(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
            nn.ReLU(),
        )
    
    def forward(self, x: Tensor):
        return self.mlp(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d,
        num_heads,
        block_size,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(d) # TODO check
        self.mha = MultiHeadAttentionBlock(num_heads, d // num_heads, block_size)

        self.ln2 = nn.LayerNorm(d) # TODO check
        self.ffwd = FeedForward(d, d)

    def forward(self, x1: Tensor):
        x2 = self.mha(self.ln1(x1)) + x1
        x3 = self.ffwd(self.ln2(x2)) + x2
        return x3


class GPT(nn.Module):
    def __init__(
        self,
        d_in: int,
        dict_size: int,
        num_heads: int,
        num_blocks: int,
        block_size: int,
    ):
        super().__init__()
        self.dict_size = dict_size
        self.embedding = nn.Embedding(dict_size, d_in)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(d_in, num_heads, block_size) for _ in range(num_blocks)])
        self.decoder = nn.Linear(d_in, dict_size)

    def forward(self, x: Tensor):
        x1 = self.embedding(x)
        x2 = GPT._positional_encoding(x1.shape[1], x1.shape[2]).to(x.device) + x1
        x3 = self.transformer_blocks(x2)
        x4 = self.decoder(x3)
        return F.softmax(x4, dim=2)
    
    def forward_train(self, x: Tensor, targets: Tensor):
        logits = self.forward(x)
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #?ignore_index ?this is for padding I believe

    @staticmethod
    def _positional_encoding(l: int, dim: int) -> Tensor:
        a = []
        for i in range(1, dim + 1):
            trig_func = np.sin if i%2==0 else np.cos
            a.append(trig_func(np.arange(l) * (i / (10_000**(2 * i / dim)))))
        return torch.tensor(np.stack(a, axis=1, dtype=np.float32))

    def infer_random(self, max_len: int) -> Tensor:
        start_char = torch.tensor(torch.randint(0, self.dict_size, (1,)).item(), dtype=torch.long).to(next(self.parameters()).device)
        a = [start_char]
        for _ in range(max_len-1):
            so_far_pred = torch.stack(a).unsqueeze(0)
            char = self.forward(so_far_pred)[0][0].multinomial(1)
            a.append(char[0])
        return [i.item() for i in a]
