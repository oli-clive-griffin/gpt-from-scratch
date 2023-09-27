import torch
from model import GPT
from typing import Callable


def infinite():
  i = 0
  while True:
    yield i
    i += 1

batch_size = 64
d_in = 32 + 16
block_size = 8
train_pct = 0.8
num_block = 4
num_heads = 2

if __name__ == "__main__":
  with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

  chars = sorted(list(set(text)))
  char_to_idx = {ch: i for i, ch in enumerate(chars)}
  idx_to_char = {i: ch for i, ch in enumerate(chars)}
  encode = lambda str: [char_to_idx[ch] for ch in str]
  decode = lambda idxs: ''.join([idx_to_char[i] for i in idxs]).replace('\n', '\\n')
  
  data = torch.tensor(encode(text), dtype=torch.long)
  device = torch.device('mps')

  train = data[:int(train_pct*len(data))].to(device)

  gpt = GPT(
    d_in=d_in,
    dict_size=len(chars),
    num_heads=num_heads,
    num_blocks=num_block,
    block_size=block_size,
    device=device,
  ).to(device)

  optimizer = torch.optim.Adam(gpt.parameters(), lr=8e-4)
  batch_len = block_size * batch_size
  print(f"should do {len(train) // batch_len} batches per epoch")

  for epoch in infinite():
    for batch_idx in range(len(train) // batch_len):
      gpt.train()
      batch_x = train[batch_idx * batch_len : (batch_idx+1) * batch_len].reshape(batch_size, block_size)
      batch_y = torch.roll(batch_x, -1, dims=1)
      optimizer.zero_grad()
      logits, loss = gpt.forward_train(batch_x, batch_y)
      loss.backward()
      optimizer.step()
      print(f'epoch {epoch} batch {batch_idx} loss {loss.item()}')
      print(f'<>{decode(batch_x[0].cpu().numpy())}<>\n<>{decode(batch_y[0].cpu().numpy())}<>\n<>{decode(torch.argmax(logits[0], dim=1).cpu().numpy())}<>\n\n')
      break

    # gpt.eval()
    # sentence = gpt.infer_random(block_size // 3)
    # print(f'epoch {epoch}')
    # print(f"<>{decode(sentence)}<>\n\n")
