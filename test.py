import torch
from torch import Tensor
import numpy as np
from PIL import Image

from model import (
  add_positional_encoding,
  positional_encoding,
  attention_matrix,
  scaled_dot_product_attention,
)


print('asdf')
def green(s): return F"\033[92m{s}\033[0m"


def test_positional_encoding():
  a = positional_encoding(200, 100)
  Image.fromarray(a * 255).show()


def test_add_positional_encoding():
  batch_size = 20
  seq_lenth = 100
  d_in_out = 60
  pos_features = 40
  x = Tensor(np.random.rand(batch_size, seq_lenth, d_in_out))
  a = add_positional_encoding(x, pos_features)
  assert a.shape == (batch_size, seq_lenth, d_in_out + pos_features)
  Image.fromarray(a[0].numpy() * 255).show()


def test_attention_matrix():
  batch_size = 20
  seq_lenth = 100
  features = 60
  q = torch.rand((batch_size, seq_lenth, features))
  k = torch.rand((batch_size, seq_lenth, features))
  attn_matrix = attention_matrix(q, k)
  assert attn_matrix.shape == (batch_size, seq_lenth, seq_lenth)
  summed = attn_matrix.sum(dim=2)
  assert summed.shape == (batch_size, seq_lenth)
  assert torch.allclose(summed, torch.ones((batch_size, seq_lenth)))
  print(green('passed'))


def test_attention_matrix_mask():
  batch_size = 20
  seq_lenth = 100
  features = 3
  q = torch.rand((batch_size, seq_lenth, features))
  k = torch.rand((batch_size, seq_lenth, features))
  attn_matrix = attention_matrix(q, k, masked=True)
  batch1 = attn_matrix[0]
  neg_inf = float('-inf')
  print(batch1)
  assert batch1[0][0] != neg_inf and batch1[0][1] == neg_inf and batch1[0][2] == neg_inf
  assert batch1[1][0] != neg_inf and batch1[1][1] != neg_inf and batch1[1][2] == neg_inf
  assert batch1[2][0] != neg_inf and batch1[2][1] != neg_inf and batch1[2][2] != neg_inf
  print(green('passed'))


def test_self_attention():
  batch_size = 1
  seq_lenth = 3
  features = 2
  k = torch.tensor([[[1, 0],
                     [0, 1],
                     [-1, 0]]])
  q = torch.tensor([[[-1, 1], # cares about a combination of idx 1 and 2
                     [-1, -1], # cares about only idx 2
                     [1, -1]]]) # cares about only idx 0
  v = torch.tensor([[[8, 9],
                     [4, 5],
                     [1, 6]]])
  x = scaled_dot_product_attention(q, k, v)
  assert x.shape == (batch_size, seq_lenth, features)
  # [-1, 1], # cares about a combination of idx 1 and 2
  assert x[0][0] == ((v[:,1] + v[:,2]) / 2)
  # [-1, -1], # cares about only idx 2
  # [1, -1], # cares about only idx 0
  print(green('passed'))

if __name__ == "__main__":
  test_attention_matrix_mask()