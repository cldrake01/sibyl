import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # print(f"(DecoderLayer.forward) x.shape: {x.shape}")
        x, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        # print(f"(DecoderLayer.forward) x.shape: {x.shape}")
        x = x + self.dropout(x)
        # print(f"(DecoderLayer.forward) x.shape: {x.shape}")
        x = self.norm1(x)

        # print(f"(DecoderLayer.forward) x.shape: {x.shape}")
        # print(f"(DecoderLayer.forward) cross.shape: {cross.shape}")

        # f, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        # f = f + self.dropout(f)
        # f = self.norm2(f)

        # x = s + f

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm:
            x = self.norm(x)

        return x
