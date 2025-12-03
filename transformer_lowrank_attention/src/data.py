import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import WikiText2
from collections import Counter
import os
import pickle

# Build a tiny tokenizer/vocab for quick experiments
class SimpleVocab:
    def __init__(self, tokens, min_freq=2, unk_token='<unk>'):
        counts = Counter(tokens)
        self.itos = ['<pad>', unk_token, '<bos>', '<eos>'] + [t for t,c in counts.items() if c >= min_freq]
        self.stoi = {s:i for i,s in enumerate(self.itos)}
    def encode(self, toks):
        return [self.stoi.get(t, self.stoi['<unk>']) for t in toks]
    def decode(self, ids):
        return [self.itos[i] for i in ids]
    def __len__(self):
        return len(self.itos)

class WikiText2SeqDataset(Dataset):
    def __init__(self, split='train', seq_len=128, vocab=None, rebuild_vocab=False, vocab_path='vocab.pkl'):
        raw_iter = WikiText2(root='.data', split=split)
        tokens = []
        for line in raw_iter:
            toks = line.strip().split()
            tokens.extend(toks + ['<eos>'])
        if vocab is None or rebuild_vocab:
            self.vocab = SimpleVocab(tokens, min_freq=2)
            with open(vocab_path, 'wb') as f:
                pickle.dump(self.vocab, f)
        else:
            self.vocab = vocab
        # convert to ids
        self.ids = self.vocab.encode(tokens)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.ids) // self.seq_len - 1)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start:start + self.seq_len]
        y = self.ids[start + 1:start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def get_dataloaders(seq_len=128, batch_size=32, rebuild_vocab=False):
    # build train vocab first
    train_ds = WikiText2SeqDataset('train', seq_len=seq_len, rebuild_vocab=rebuild_vocab)
    # reuse saved vocab
    import pickle
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    val_ds = WikiText2SeqDataset('valid', seq_len=seq_len, vocab=vocab)
    test_ds = WikiText2SeqDataset('test', seq_len=seq_len, vocab=vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader, len(vocab)
