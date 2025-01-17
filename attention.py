import torch
from torch import nn
from torch.nn import functional as F
import math


# building the attention mechanism
class SelfAttention(nn.Module):
    def __init__(
        self, n_head: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_head = n_head
        self.d_embed = d_embed
        self.d_head = d_embed // n_head

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_head, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).tranpose(1, 2)
        k = k.view(interim_shape).tranpose(1, 2)
        v = v.view(interim_shape).tranpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output
