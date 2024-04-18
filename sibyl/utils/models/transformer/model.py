from typing import Collection

import torch
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        d_layers: int,
        d_model: int = 512,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        A decoder-only transformer model.
        """
        super(Transformer, self).__init__()
        self.linear = nn.Linear(X.size(1), d_model)
        self.embedding = Rotary(d_model)
        self.decoder = Decoder(
            tuple(
                DecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(d_layers)
            ),
        )
        self.projection = nn.Linear(d_model, y.size(1), bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x.mT)
        x = self.embedding(x)
        x = self.decoder(x)
        x = self.projection(x.squeeze().mT)
        x = x.unsqueeze(0).permute(0, 2, 1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        layers: Collection[nn.Module],
    ):
        super(Decoder, self).__init__()
        self.norm = nn.LayerNorm(len(layers))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        self.temporal_cross_attention = TemporalCrossAttention(
            d_model, n_heads, dropout
        )
        self.feature_cross_attention = FeatureCrossAttention(d_model, n_heads, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.self_attention(x)
        x += self.dropout(x)
        x = self.norm1(x)

        x, _ = self.feature_cross_attention(x)
        x += self.dropout(x)
        x = self.norm2(x)

        x = self.conv1(x.mT)
        x = self.conv2(x).mT
        x = self.norm3(x).unsqueeze_(0).permute(2, 1, 0)

        return x


class SelfAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, output_attention: bool = False
    ):
        super(SelfAttention, self).__init__()
        self.output_attention = output_attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )
        self.mask = torch.tril(torch.zeros((d_model, d_model)), 1).bool()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        x = x.mT
        out, attention = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self.mask,
            need_weights=self.output_attention,
        )
        return out, attention if self.output_attention else None


class TemporalCrossAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, output_attention: bool = False
    ):
        super(TemporalCrossAttention, self).__init__()
        self.output_attention = output_attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor | None]:
        out, attention = self.attn(
            query=x, key=y, value=y, need_weights=self.output_attention
        )
        attention = attention if self.output_attention else None
        return out, attention


class FeatureCrossAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float, output_attention: bool = False
    ):
        """
        Model cross-attention for each feature, as opposed to the entire sequence.
        """
        super(FeatureCrossAttention, self).__init__()
        self.output_attention = output_attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        x_ = torch.zeros_like(x)
        attention_ = None
        for i in range(x.size(1)):
            x_i = torch.zeros_like(x)
            x_i[..., i] = x[..., i]
            x_i_prime = torch.where(x_i == 0, x, 0)
            out, _ = self.attn(
                query=x_i,
                key=x_i_prime,
                value=x_i_prime,
                need_weights=self.output_attention,
            )
            x_ += out
        return x_, attention_ if self.output_attention else None

    # def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
    #     seq_len, _ = x.size()
    #
    #     # Create a mask for zero elements
    #     mask = (x == 0).all(dim=-1, keepdim=True).unsqueeze(-1)
    #
    #     # Create x_i_prime by masking zero elements
    #     x_i_prime = x.masked_fill(mask, 0)
    #
    #     # Duplicate x_i_prime along the sequence length dimension
    #     x_i_prime_repeat = x_i_prime.unsqueeze(1).repeat(1, seq_len, 1, 1)
    #
    #     # Duplicate x along the sequence length dimension
    #     x_repeat = x.unsqueeze(2).repeat(1, 1, seq_len, 1)
    #
    #     # Apply attention mechanism
    #     out, _ = self.attn(
    #         query=x_repeat,
    #         key=x_i_prime_repeat,
    #         value=x_i_prime_repeat,
    #         need_weights=self.output_attention,
    #     )
    #
    #     # Mask out the padding
    #     out.masked_fill_(mask.repeat(1, seq_len, 1, 1), 0)
    #
    #     # Sum along the sequence length dimension
    #     x_ = out.sum(dim=1)
    #
    #     return x_, _ if self.output_attention else None


class Rotary(torch.nn.Module):
    def __init__(self, d_model: int):
        super(Rotary, self).__init__()
        self.d_model = d_model

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        # Remove the batch dimension, as, otherwise, the array slicing will not work
        x = x.squeeze()

        # Compute a tensor of alternating values from x
        x_alternating = torch.zeros_like(x)
        x_alternating[..., ::2] = -x.abs()[..., ::2]
        x_alternating[..., 1::2] = x.abs()[..., 1::2]

        # Θ = {θ_i = 10,000^{ ( −2(i−1) ) / d } , i ∈ [1, 2, ..., d/2]}
        theta = 10_000 ** (-2 * (torch.arange(1, x.size(-1) + 1) - 1) / x.size(-1))

        # Compute the cosine and sine tensors
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        x_cos = torch.inner(x, cos)
        x_alternating_sin = torch.inner(x_alternating, sin)

        # Add the two resultant tensors to obtain the final tensor
        r_theta = x_cos + x_alternating_sin

        return x * r_theta.unsqueeze(0).mT
