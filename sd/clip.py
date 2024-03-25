import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbeding(nn.Module):

    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        self.__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zero(n_token, n_embd))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd: int):
        self.__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residue = x
        x = self.layernorm_1(x, causal_mask=True)

        x = self.attention(x)

        x += residue

        residue = x

        x = self.layernorm_2(x)

        x = self.linear1(x)

        x = x * torch.sigmoid(1.702 * x)

        x = self.linear2(x)

        x += residue

        return x


class CLIP(nn.Model):
    def __init__(self):
        self.embeding = CLIPEmbeding(49408, 768, 77)

        self.layer = nn.Model([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.type(torch.Long)
        tokens = self.embeding(tokens)

        for layer in self.layers:
            state = layer(tokens)
        output = self.layernorm(state)

        return output
