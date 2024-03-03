import torch
from torch import nn

from sibyl.utils.models.ring.layers.attn import AttentionLayer, FullAttention
from sibyl.utils.models.ring.layers.decoder import Decoder, DecoderLayer
from sibyl.utils.models.ring.layers.encoder import Encoder, EncoderLayer, ConvLayer
from sibyl.utils.models.ring.ring_attention import RingAttention, RingRotaryEmbedding


class Ring(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.01,
        attn="prob",
        embed="fixed",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        encoder=False,
        mix=True,
    ):
        super(Ring, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Assuming enc_in and dec_in are the dimensions of x and x_mark respectively
        # self.enc_embedding = PositionalEmbedding(d_model)
        # self.dec_embedding = PositionalEmbedding(d_model)
        self.enc_embedding = RingRotaryEmbedding(
            dim=d_model,
            ring=True,
            striped=True,
        )
        self.dec_embedding = RingRotaryEmbedding(
            dim=d_model,
            ring=True,
            striped=True,
        )

        # Attention
        Attn = RingAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attn(
                        dim=d_model,
                        striped_ring_attn=True,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        ) if encoder else None
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    Attn(
                        dim=d_model,
                        striped_ring_attn=True,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
            self,
            x_enc,
            x_dec,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
    ):
        attentions = None
        enc_out = self.enc_embedding(x_enc.size(1))
        dec_out = self.dec_embedding(x_dec.size(1))
        if self.encoder:
            enc_out, attentions = self.encoder(enc_out, attn_mask=enc_self_mask)
            dec_out = self.decoder(
                dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
            )
        else:
            dec_out = self.decoder(
                dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
            )
        dec_out = self.projection(dec_out).transpose(1, 2)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attentions
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
