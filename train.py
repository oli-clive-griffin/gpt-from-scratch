import wandb
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import GPT
import itertools
from typing import Iterable
import os
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenizers import ByteLevelBPETokenizer


class Corpus:
    def __init__(self, text: str):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_to_idx =  { ch: i for i, ch in enumerate(self.chars) }

        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(files="input.txt", vocab_size=512)
        self.dict_size = 512
        
    def encode(self, s: str):
        encoding = self.tokenizer.encode(s)
        vocab = self.tokenizer.get_vocab()
        return [vocab[tok] for tok in encoding.tokens]

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

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        return sequence[:-1], sequence[1:]


def create_data_loaders(text_enc: list[int], block_size: int, train_pct: float, batch_size: int):
    whole_dataset = TextDataset(text_enc, block_size)

    train_size = int(train_pct * len(whole_dataset))
    val_size = len(whole_dataset) - train_size
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size, sampler=DistributedSampler(val_dataset))

    return train_loader, val_loader


local_rank = int(os.getenv("LOCAL_RANK"))
torch.cuda.set_device(local_rank)

torch.distributed.init_process_group(backend="nccl")

if __name__ == "__main__":

    file_path = 'input.txt'
    block_size = 512
    train_pct = 0.9
    batch_size = 128
    num_block = 6
    num_heads = 4
    d_in = 64 * num_heads
    dropout = 0.3

    log_wandb = 'WANDB_KEY' in os.environ

    device = torch.device('cuda')

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        corpus = Corpus(text)

    train_loader, val_loader = create_data_loaders(corpus.encode(corpus.text), block_size, train_pct, batch_size)
    model = GPT(d_in, corpus.dict_size, num_heads, num_block, block_size, dropout).to(device)
    model = DDP(model)
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
    
    lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

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
            lrs.step()

            # if batch_idx % 100 == 99:
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
    