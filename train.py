import torch
from model import GPT


batch_size = 20
d_in = 64
d_pos_enc = 64
block_size = 256
seq_length = 100

if __name__ == "__main__":
  with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  
  chars = sorted(list(set(text)))

  char_to_idx = {ch: i for i, ch in enumerate(chars)}
  idx_to_char = {i: ch for i, ch in enumerate(chars)}
  encode = lambda str: [char_to_idx[ch] for ch in str]
  decode = lambda idxs: ''.join([idx_to_char[i] for i in idxs])

  data = torch.tensor(encode(text), dtype=torch.long)
  train = data[:int(0.8*len(data))]
  test = data[int(0.8*len(data)):]

  gpt = GPT(d_in, d_pos_enc, len(chars), num_heads=4, num_blocks=4, block_size=block_size)
  optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-3)
  batch_len = block_size * batch_size

  for epoch in range(10):
    for batch_idx in train // batch_len:
      batch_x = train[batch_idx * batch_len : (batch_idx+1) * batch_len].reshape(batch_size, block_size)
      batch_y = torch.roll(batch_x, -1).view(batch_len)

      optimizer.zero_grad()
      logits, loss = gpt.forward_train(batch_x, batch_y)
      breakpoint()


      print(F"{epoch=} {loss.item()=}")
      loss.backward()
      optimizer.step()
