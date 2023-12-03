import math
import torch
import torch.nn as nn

from src.utils import BOS_ID, EOS_ID, text2ids, ids2text


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 400):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_len: int, d_model: int, nhead: int, dim_feedforward: int, nlayers: int, max_len: int = 400, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.head = nn.Linear(d_model, vocab_len)

        self._init_weights()

    def _init_weights(self):
        bound = 0.1
        self.embedding.weight.data.uniform_(-bound, bound)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-bound, bound)

    def forward(self, src, src_mask=None):
        src_padding_mask = src == 0  # no effect
        src = src.transpose(0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0]).to(src.device)
        output = self.transformer_encoder(src, src_mask, src_padding_mask)
        output = self.linear(output)
        output = output.transpose(0, 1)
        return {"logits": output}

    @torch.inference_mode()
    def inference(self, prefix: str = "", temp: float = 1.0) -> str:
        self.eval()

        tokens = [BOS_ID] + text2ids(prefix)
        tokens = torch.tensor(tokens).to(next(self.parameters()).device)

        logits = self.forward(tokens.unsqueeze(0))["logits"].transpose(1, 2) / temp
        new_tokens = torch.distributions.categorical.Categorical(logits=logits[:, :, -1]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=0)

        while tokens.shape[0] < self.max_len:
            if new_tokens.item() == EOS_ID:
                break

            logits = self.forward(tokens.unsqueeze(0))["logits"].transpose(1, 2) / temp
            new_tokens = torch.distributions.categorical.Categorical(logits=logits[:, :, -1]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=0)

        return ids2text(tokens.squeeze())
