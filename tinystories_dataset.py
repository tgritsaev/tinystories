import os
import glob

import sentencepiece as spm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items):
    pad_id = 0
    result_batch = {}
    result_batch["input"] = pad_sequence([item["input"] for item in dataset_items], batch_first=True, padding_value=pad_id)
    result_batch["padding_mask"] = result_batch["input"] == pad_id
    return result_batch


def move_batch_to_device(device, **batch):
    for key in ["input", "padding_mask"]:
        batch[key] = batch[key].to(device)


class TinyStoriesDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer_path: str, limit: int = None):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        print("Scanning input dir...")
        self.index = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.index = self.index[:limit]
        print("Dataset has been created.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return {"input": torch.tensor(np.load(self.index[index]))}
