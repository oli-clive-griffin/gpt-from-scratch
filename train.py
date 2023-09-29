import wandb
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import GPT
import itertools
from typing import Iterable
import os


class Corpus:
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_to_idx =  { ch: i for i, ch in enumerate(self.chars) }
        self.dict_size = len(self.chars)

    def encode(self, s: str):
        return [self.char_to_idx[ch] for ch in s]

    def decode(self, xs: Iterable[int]):
        return ''.join([self.chars[x] for x in xs])


class TextDataset(Dataset[torch.Tensor]):
    def __init__(self, text: list[int], seq_len: int):
        self.sequences = [
            torch.tensor(text[i:i+seq_len], dtype=torch.long) 
            for i in range(0, len(text) - seq_len + 1, seq_len)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sequence = self.sequences[idx]
        return sequence[:-1], sequence[1:]


def create_data_loaders(text_enc: list[int], block_size: int, train_pct: float, batch_size: int):
    whole_dataset = TextDataset(text_enc, block_size)

    train_size = int(train_pct * len(whole_dataset))
    val_size = len(whole_dataset) - train_size
    train_data, val_data = random_split(whole_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":

    file_path = 'input.txt'
    block_size = 128
    train_pct = 0.9
    batch_size = 32
    d_in = 64 * 6
    num_block = 6
    num_heads = 6
    log_wandb = 'WANDB_KEY' in os.environ
    dropout = 0.3

    device = torch.device('mps')

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        corpus = Corpus(text)

    train_loader, val_loader = create_data_loaders(corpus.encode(corpus.text), block_size, train_pct, batch_size)
    model = GPT(d_in, corpus.dict_size, num_heads, num_block, block_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    if log_wandb:
        wandb.init(
            config={
                'batch_size': batch_size,
                'd_in': d_in,
                'block_size': block_size,
                'train_pct': train_pct,
                'num_block': num_block,
                'num_heads': num_heads,
                'dropout': dropout,
            },
        )

    for epoch in itertools.count():
        if log_wandb:
            wandb.log({ 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], })

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader): 
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            model.train()
            optimizer.zero_grad()
            logits, loss = model.forward(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch} batch {batch_idx} loss {loss.item()}')

            if log_wandb and batch_idx % 10 == 0:
                wandb.log({ 'train_loss': loss.item() })

        for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            model.eval()
            logits, loss = model.forward(batch_x, batch_y)
            print(f'epoch {epoch} val batch {batch_idx} loss {loss.item()}')

            if log_wandb and batch_idx % 10 == 0:
                wandb.log({ 'val_loss': loss.item() })

        print(f'epoch {epoch}')
    