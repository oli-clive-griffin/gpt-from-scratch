import torch
from model import GPT
from typing import Callable


def infinite():
  i = 0
  while True:
    yield i
    i += 1

batch_size = 128
d_in = 48
d_pos_enc = 16
block_size = 128
train_pct = 0.8

if __name__ == "__main__":
  with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

  # text = 'ab' * 100000
  
  chars = sorted(list(set(text)))

  char_to_idx = {ch: i for i, ch in enumerate(chars)}
  idx_to_char = {i: ch for i, ch in enumerate(chars)}
  encode = lambda str: [char_to_idx[ch] for ch in str]
  decode = lambda idxs: ''.join([idx_to_char[i] for i in idxs]).replace('\n', '\\n')

  data = torch.tensor(encode(text), dtype=torch.long)
  device = torch.device('mps')

  train = data[:int(train_pct*len(data))].to(device)
  test = data[int(train_pct*len(data)):].to(device)

  gpt = GPT(
    d_in,
    d_pos_enc,
    len(chars),
    num_heads=4,
    num_blocks=4,
    block_size=block_size,
    device=device,
  ).to(device)

  optimizer = torch.optim.Adam(gpt.parameters(), lr=5e-5)
  batch_len = block_size * batch_size

  for epoch in infinite():
    for batch_idx in range(len(train) // batch_len):
      batch_x = train[batch_idx * batch_len : (batch_idx+1) * batch_len].reshape(batch_size, block_size)
      batch_y = torch.roll(batch_x, -1).view(batch_len)
      optimizer.zero_grad()

      logits, loss = gpt.forward_train(batch_x, batch_y)
      loss.backward()
      optimizer.step()
      if batch_idx % 10 == 0:
        print(f'epoch {epoch} batch {batch_idx} loss {loss.item()}')
        print(decode(torch.argmax(logits, dim=1).flatten().tolist())[:100])
