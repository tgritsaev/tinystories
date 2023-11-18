import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)  # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition
        tmp = torch.arange(0, max_len).unsqueeze(1) / torch.pow(10000, torch.arange(0, embed_dim, 2) / embed_dim)
        pe[0, :, ::2] = torch.sin(tmp)
        pe[0, :, 1::2] = torch.cos(tmp)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_len, d_model, num_layers, nhead, dropout, max_len=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        block = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers)
        self.head = nn.Linear(d_model, vocab_len)

    def forward(self, input, padding_mask, **kwargs):
        x = self.embedding(input)
        x = self.positional_encoding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(input.shape[1], device=input.device)
        seq = self.encoder(x, mask, padding_mask, is_causal=True)

        return {"logits": self.head(seq)}
