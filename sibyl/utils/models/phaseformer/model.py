import torch
from torch import nn, Tensor


class Print(nn.Module):
    def __init__(self, letter: str):
        super(Print, self).__init__()
        self.letter = letter

    def forward(self, i: int, query: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        # print(f"{x = }")
        # print(f"{self.letter} \t {x.size() = }")
        print(f"{self.letter} \t {i = }")
        return query, query


class Phaseformer(nn.Module):
    def __init__(
        self,
        dimensions: int,
        duration: int,
        feature_rank: int,
        temporal_rank: int,
        dropout: float,
        num_heads: int | None = None,
    ):
        super(Phaseformer, self).__init__()
        self.dimensions = dimensions
        self.duration = duration
        self.feature_rank = feature_rank
        self.temporal_rank = temporal_rank
        self.dropout = dropout
        self.num_heads = num_heads or dimensions * duration
        self.conv_a = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.dimensions,
                    out_channels=self.dimensions,
                    kernel_size=i,
                ),
                nn.GELU(),
                nn.LayerNorm([1, self.dimensions, self.duration - i + 1]),
                nn.Dropout(dropout),
            )
            for i in range(1, self.duration + 1)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=sum(range(1, self.duration + 1)),
            num_heads=sum(range(1, self.duration + 1)),
            dropout=self.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        # print(x.size())
        x = tuple(conv(x) for conv in self.conv_a)
        # print(x.__repr__())
        x = torch.cat(x, dim=self.temporal_rank)
        # x = x.rename("batch", "feature", "temporal")
        # In order to explicitly examine how differing indicators affect each other,
        # we apply cross-attention to each of the input tensor's features.
        x = torch.sum(
            torch.stack(
                tuple(
                    self.attention(
                        query=x,
                        key=x,
                        value=x[:, i, :] + torch.zeros_like(x),
                        need_weights=False,
                    )[0]
                    for i in range(x.size(self.feature_rank))
                ),
            ),
            dim=0,
        )
        # We now have to undo the convolutional operation.
        x = x.unfold(self.temporal_rank, self.duration, self.duration * 4).mean(-1)
        # torch.set_printoptions(profile="full")
        # print(x)
        x = x.permute(0, 2, 1)
        # print(x.size())
        return x
