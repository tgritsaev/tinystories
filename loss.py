import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, pad_id=0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, input, logits, **kwargs):
        return self.loss_fn(logits[:, :-1].transpose(1, 2), input[:, 1:])
