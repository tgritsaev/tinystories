import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import PAD_ID, BOS_ID, EOS_ID


def collate_fn(items):
    batch = {}
    batch["src"] = pad_sequence([item["src"] for item in items], True, PAD_ID)
    return batch


class TinyStoriesDataset(Dataset):
    def __init__(self, part: str, data_path: str, max_len: int = None, limit: int = None):
        super().__init__()
        self.max_len = max_len
        print(f"Loading data from {part}...")
        self.idxs = torch.from_numpy(np.load(f"{data_path}/{part}_idxs.npy")).to(torch.int64)[:limit]
        self.data = torch.from_numpy(np.load(f"{data_path}/{part}.npy")).to(torch.int16)
        print(f"Dataset has been created, dataset info:")
        print(f"size:\t{self.idxs.shape[0]}\nmin:\t{self.data.min()}\nmax:\t{self.data.max()}\nshape:\t{self.data.shape}")

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, idx):
        begin, end = self.idxs[idx]
        if self.max_len:
            end = min(end, begin + self.max_len)
        src = [BOS_ID] + list(self.data[begin : end + 1]) + [EOS_ID]
        return {"src": torch.tensor(src)}
