import torch
from torch import nn, Tensor
import numpy as np


class AttentionBlock(nn.Module):
  def __init__(
    self,
    d_in = 16,
    d_k = 16,
    d_v = 32,
  ):
    super().__init__()
    self.q = nn.Linear(in_features=d_in, out_features=d_k)
    self.k = nn.Linear(in_features=d_in, out_features=d_k)
    self.v = nn.Linear(in_features=d_in, out_features=d_v)

  def forward(self, x: Tensor, masked = False):
    Q = self.q(x)
    K = self.k(x)
    V = self.v(x)
    x = scaled_dot_product_self_attention(Q, K, V, masked)
    return x


class MultiHeadAttentionBlock(nn.Module):
  def __init__(
    self,
    num_heads = 4,
    d_per_head = 16,
  ):
    super().__init__()
    self.d_per_head = d_per_head
    self.num_heads = num_heads
    self.heads = [AttentionBlock(d_per_head, d_per_head, d_per_head) for _ in range(num_heads)]

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


class TransformerBlock(nn.Module):

  def __init__(
    self,
    d = 8,
    num_heads = 4,
  ):
    super().__init__()
    self.num_heads = num_heads

    self.ln1 = nn.LayerNorm(d) # TODO check
    self.mha = MultiHeadAttentionBlock(num_heads, d // num_heads)

    self.ln2 = nn.LayerNorm(d) # TODO check
    self.mlp = nn.Linear(in_features=d, out_features=d)

  def forward(self, x: Tensor):
    x = self.mha(self.ln1(x)) + x
    x = self.mlp(self.ln2(x)) + x
    return x


class Sentimentalist(nn.Module):
  def __init__(self, d_in: int, d_pos_enc: int, num_classes: int):
    super().__init__()
    d = d_in + d_pos_enc
    self.d_pos_enc = d_pos_enc
    self.transformer_blocks = [TransformerBlock(d) for _ in range(10)]
    self.mlp = nn.Linear(in_features=d, out_features=num_classes)
  
  def forward(self, x: Tensor):
    x = add_positional_encoding(x, self.d_pos_enc)

    for block in self.transformer_blocks:
      x = block(x)

    x = self.mlp(x[:,-1,:])

    return nn.functional.softmax(x, dim=1)


def scaled_dot_product_self_attention(q: Tensor, k: Tensor, v: Tensor, masked = False) -> Tensor: # shape (batch_size, seq_length, dim_v)
  assert len(q.shape) == 3 and len(k.shape) == 3, 'expected both q and k to be 3d'
  attn_matrix = attention_matrix(q, k, masked)
  d_k = torch.Tensor([k.shape[2]])
  return (attn_matrix @ v) / torch.sqrt(d_k)


def attention_matrix(q: Tensor, k: Tensor, masked = False):
    attn_matrix = torch.matmul(q, k.transpose(dim0=1, dim1=2)).softmax(dim=2)
    return torch.tril(attn_matrix) if masked else attn_matrix


def positional_encoding(l: int, dim: int) -> np.ndarray:
  a = []
  for i in range(1, dim + 1):
    trig_func = np.sin if i%2==0 else np.cos
    a.append(trig_func(np.arange(l) * (i / (10_000**(2 * i / dim)))))
  return np.stack(a, axis=1)


def add_positional_encoding(x: Tensor, dim: int) -> Tensor:
  assert len(x.shape) == 3 # (batch_size, seq_lenth, d_in_out)

  pos_enc = positional_encoding(x.shape[1], dim)  # seq_lenth
  batch_size = x.shape[0]
  expanded = np.stack([pos_enc for _ in range(batch_size)], dtype=np.float32)
  pos_tensor = torch.from_numpy(expanded)
  return torch.cat((x, pos_tensor), dim=2) # cat on feature dim


if __name__ == "__main__":
  batch_size = 20
  d_in = 64
  d_enc = 64
  d = d_in + d_enc
  seq_length = 100
  input = Tensor(np.random.randn(batch_size, seq_length, d_in))
  s = Sentimentalist(d_in, d_enc, 3)
  x = s.forward(input)
