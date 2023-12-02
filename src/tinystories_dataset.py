import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import PAD_ID, BOS_ID, EOS_ID


def collate_fn(items):
    batch = {}
    batch["src"] = pad_sequence([item["src"] for item in items], True, PAD_ID)
    # batch["mask"] = batch["x"] == PAD_ID
    return batch


def move_batch_to_device(device, **batch):
    for key in ["src"]:  # , "mask"
        batch[key] = batch[key].to(device)


class TinyStoriesDataset(Dataset):
    def __init__(self, part: str, data_path: str, limit: int = None):
        super().__init__()
        print(f"Loading data from {part}...")
        self.idxs = np.load(f"{data_path}/{part}_idxs.npy").astype(np.int16)[:limit]
        self.data = np.load(f"{data_path}/{part}.npy").astype(np.int64)
        print(f"Dataset has been created, data.shape: {self.data.shape}")

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, idx):
        begin, end = self.idxs[idx]
        return {"src": torch.tensor([BOS_ID] + self.data[begin : end + 1] + [EOS_ID])}
