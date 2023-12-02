import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import PAD_ID, BOS_ID, EOS_ID, text2ids, ids2text


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
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
    def __init__(
        self,
        nlayers: int,
        nhead: int,
        vocab_len: int,
        d_model: int,
        dim_feedforward: int,
        activation=F.leaky_relu,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_len, d_model, PAD_ID)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.head = nn.Linear(d_model, vocab_len)
        self._init_weights()

    def _init_weights(self):
        for param in self.parameters():
            if param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)
            else:
                nn.init.uniform_(param)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        print(f"src.device: {src.device}")
        x = torch.sqrt(torch.tensor(self.d_model, device=src.device)) * self.embedding(src)
        x = self.positional_encoding(x)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[0]).to(x.device)
        x = self.transformer_encoder(x, mask)
        return {"logits": self.head(x)}

    @torch.inference_mode()
    def inference(self, prefix: str = "", temp: float = 1.0) -> str:
        self.eval()
        device = next(self.parameters()).device

        # encode prefix
        tokens = [BOS_ID] + text2ids(prefix)
        tokens = torch.tensor(tokens).to(device)

        # generate hidden for prefix
        logits = self.forward(tokens.unsqueeze(1))
        logits = logits.transpose(0, 1).transpose(1, 2)
        logits /= temp

        # sample new token from logits
        new_tokens = torch.distributions.categorical.Categorical(logits=logits[:, :, -1]).sample()
        tokens = torch.cat([tokens, new_tokens], dim=0)

        # 2 stopping conditions: reaching max len or getting <eos> token
        while tokens.shape[0] < self.max_len:
            if new_tokens.item() == EOS_ID:
                break

            # process newly obtained token
            logits = self.forward(tokens.unsqueeze(1))
            logits = logits.transpose(0, 1).transpose(1, 2)
            logits /= temp

            # sample the next token from logits
            new_tokens = torch.distributions.categorical.Categorical(logits=logits[:, :, -1]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=0)

        return {"text": ids2text(tokens.squeeze())}
