import torch.nn as nn
import torch.nn.functional as F

from src.utils import PAD_ID


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    def forward(self, logits, src, **kwargs):
        # loss(B x vocab_size x max_len, B x max_len)
        return self.loss_fn(logits.transpose(1, 2), src[:, 1:])
