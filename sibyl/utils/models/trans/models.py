import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def create_padding_mask(seq):
    # Creating a mask for padding tokens
    return (seq == 0).transpose(0, 1)


def create_look_ahead_mask(size):
    # This mask hides future positions - for decoder
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask


def create_masks(src, tgt):
    # Encoder padding mask
    src_mask = create_padding_mask(src)

    # Decoder mask
    tgt_mask = create_padding_mask(tgt)

    # Look ahead mask for target
    size = tgt.size(1)  # size of target sequence
    look_ahead_mask = create_look_ahead_mask(size)

    # Combine padding mask and look ahead mask for target
    combined_mask = torch.maximum(tgt_mask, look_ahead_mask)

    return src_mask, combined_mask


# Embedding the input sequence
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)


# The positional encoding vector
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * i + 1) / embedding_dim))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(x.device)
        # Add the positional encoding vector to the embedding vector
        x = x + pe
        x = self.dropout(x)
        return x


# Self-attention layer
class SelfAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        def forward(self, query, key, value, mask=None):
            key_dim = key.size(-1)
            attn = torch.matmul(query / np.sqrt(key_dim), key.transpose(2, 3))
            if mask is not None:
                mask = mask.unsqueeze(1)
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(torch.softmax(attn, dim=-1))
            output = torch.matmul(attn, value)

            return output
        """
        print(f"(SelfAttention.forward) query: {query.shape}")
        print(f"(SelfAttention.forward) key: {key.shape}")
        print(f"(SelfAttention.forward) value: {value.shape}")

        def attention(q, k, v, i, m=None):
            print(f"(SelfAttention.forward.attention) query: {q}")
            print(f"(SelfAttention.forward.attention) key: {k}")
            print(f"(SelfAttention.forward.attention) value: {v}")
            print(f"(SelfAttention.forward.attention) i: {i}")

            v_i = torch.zeros_like(v)[..., i] + v[..., i]

            key_dim = k.size(-1)
            attn_i = torch.matmul(q / np.sqrt(key_dim), k.transpose(2, 3))
            if m is not None:
                m = m.unsqueeze(1)
                attn_i = attn_i.masked_fill(m == 0, -1e9)
            attn_i = self.dropout(torch.softmax(attn_i, dim=-1))
            o_i = torch.matmul(attn_i, v_i)

            return o_i, attn_i

        # Calculate the attention weights for each dimension of the value vector
        output_list = [
            attention(query, key, value, i, mask)[0] for i in range(value.size(-1))
        ]

        # Concatenate the output
        output = torch.sum(torch.cat(output_list, dim=-1), dim=-1)

        return torch.matmul(output, value)


# Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        # The number of heads
        self.num_heads = num_heads
        # The dimension of each head
        self.dim_per_head = embedding_dim // num_heads
        # The linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        # Apply the linear projections
        batch_size = query.size(0)
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        # Reshape the input
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(
            1, 2
        )
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(
            1, 2
        )
        # Calculate the attention
        scores = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        )
        # Apply the linear projection
        output = self.out(output)
        return output


# Norm layer
class Norm(nn.Module):
    def __init__(self, embedding_dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(x)


# Transformer encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        # Add and Muti-head attention
        x = x + self.dropout1(self.self_attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x


# Transformer decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    def forward(self, x, target_mask):
        """
        def forward(self, x, memory, source_mask, target_mask):
            x2 = self.norm1(x)
            x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
            x2 = self.norm2(x)
            x = x + self.dropout2(self.encoder_attention(x2, memory, memory, source_mask))
            x2 = self.norm3(x)
            x = x + self.dropout3(self.feed_forward(x2))
            return x

        :param x:
        :param target_mask:
        :return:
        """
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x2, x2, x2, target_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))
        return x


# Encoder transformer
class Encoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, num_heads, 2048, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, source, source_mask):
        # Embed the source
        x = self.embedding(source)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, source_mask)
        # Normalize
        x = self.norm(x)
        return x


# Decoder transformer
class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embedding_dim, num_heads, 2048, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(self, target, target_mask):
        """
        def forward(self, target, memory, source_mask, target_mask):
            # Embed the source
            x = self.embedding(target)
            # Add the position embeddings
            x = self.position_embedding(x)
            # Propagate through the layers
            for layer in self.layers:
                x = layer(x, memory, source_mask, target_mask)
            # Normalize
            x = self.norm(x)
            return x

        :param target:
        :param target_mask:
        :return:
        """
        # Embed the source
        x = self.embedding(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, target_mask)
        # Normalize
        x = self.norm(x)
        return x


# Transformers
class Transformer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        seq_len,
        out_len,
        embedding_dim,
        num_heads,
        num_layers,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.source_vocab_size = enc_in
        self.target_vocab_size = dec_in
        self.source_max_seq_len = seq_len
        self.target_max_seq_len = out_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # self.encoder = Encoder(
        #     enc_in,
        #     embedding_dim,
        #     seq_len,
        #     num_heads,
        #     num_layers,
        #     dropout,
        # )
        self.decoder = Decoder(
            dec_in,
            embedding_dim,
            out_len,
            num_heads,
            num_layers,
            dropout,
        )
        self.final_linear = nn.Linear(embedding_dim, dec_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target, source_mask, target_mask):
        # Encoder forward pass
        # memory = self.encoder(source, source_mask)
        # Decoder forward pass
        # output = self.decoder(target, memory, source_mask, target_mask)
        output = self.decoder(target, target_mask)
        # Final linear layer
        output = self.dropout(output)
        output = self.final_linear(output)
        return output

    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)

    def make_target_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (
            1
            - torch.triu(
                torch.ones((1, len_target, len_target), device=target_ids.device),
                diagonal=1,
            )
        ).bool()
        return subsequent_mask
