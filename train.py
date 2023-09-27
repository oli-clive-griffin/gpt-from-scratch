import torch
from model import GPT


def infinite():
  i = 0
  while True:
    yield i
    i += 1

batch_size = 32
d_in = 128
block_size = 64
train_pct = 0.8
num_block = 6
num_heads = 4

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

  model = GPT(
    d_in=d_in,
    dict_size=len(chars),
    num_heads=num_heads,
    num_blocks=num_block,
    block_size=block_size,
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
  batch_len = block_size * batch_size
  batches = len(train) // batch_len
  for epoch in infinite():
    padding = len(train) % batch_len

    x = train[:-padding].view(batches, batch_size, block_size)[torch.randperm(batches)]
    y = torch.roll(x, -1, dims=-1)

    for batch_idx in range(batches):
      batch_x = x[batch_idx]
      batch_y = y[batch_idx]
      
      model.train()
      optimizer.zero_grad()
      logits, loss = model.forward(batch_x, batch_y)
      loss.backward()
      optimizer.step()
      print(f'epoch {epoch} batch {batch_idx} loss {loss.item()}')
      if batch_idx % 100 == 0:
        print(f'<>{decode(batch_x[0].cpu().numpy())}<>\n<>{decode(batch_y[0].cpu().numpy())}<>\n<>{decode(torch.argmax(logits[0], dim=1).cpu().numpy())}<>\n\n')

    model.eval()
    sentence = model.infer_random(block_size // 3)
    print(f'epoch {epoch}')
    print(f"<>{decode(sentence)}<>\n\n")
