from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from sibyl.utils.models.dimformer.masking import TriangularCausalMask


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

            # Create a mask with ones at the i-th position in the third dimension and zeros elsewhere
            mask = torch.zeros_like(v)
            mask[..., i, :] = 1

            # Apply the mask to v
            v_i = v * mask

            print(f"(SelfAttention.forward.attention) v_i: {v_i}")
            print(f"(SelfAttention.forward.attention) v_i.shape: {v_i.shape}")

            key_dim = k.size(-1)
            attn_i = torch.matmul(q / np.sqrt(key_dim), k.transpose(2, 3))
            if m is not None:
                m = m.unsqueeze(1)
                attn_i = attn_i.masked_fill(m == 0, -1e9)
            attn_i = self.dropout(torch.softmax(attn_i, dim=-1))
            o_i = torch.matmul(attn_i, v_i)

            return o_i, attn_i

        # Calculate the attention weights for each dimension of the value vector

        dim = -2

        output_list = [
            attention(query, key, value, i, mask)[0] for i in range(value.size(dim))
        ]

        # Concatenate the output
        output = torch.sum(torch.cat(output_list, dim=dim), dim=dim)

        return output, None


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
        scores, attn = self.self_attention(query, key, value, mask)
        # Reshape the output
        output = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        )
        # Apply the linear projection
        output = self.out(output)
        return output, attn


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        """
        Implements Full Attention mechanism.

        :param mask_flag: If True, masks future positions to enforce causality.
        :param factor: Factor by which to reduce the number of keys.
        :param scale: Scale factor for the attention scores.
        :param attention_dropout: Dropout rate for attention weights.
        :param output_attention: If True, outputs the attention weights.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass for Full Attention.

        :param queries: Queries.
        :param keys: Keys.
        :param values: Values.
        :param attn_mask: Attention mask.

        :returns: A tuple containing the output of the attention mechanism and the attention weights.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False
    ):
        """
        A layer that wraps an attention mechanism.

        :param attention: The attention module to be used.
        :param d_model: The dimensionality of the model.
        :param n_heads: Number of attention heads.
        :param d_keys: Size of the key vectors. Defaults to d_model/n_heads.
        :param d_values: Size of the value vectors. Defaults to d_model/n_heads.
        :param mix: If True, mixes the output.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.d_model = d_model
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dim_per_head = d_model // n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        # Apply the linear projections
        batch_size = queries.size(0)
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        # Reshape the input
        queries = queries.view(
            batch_size, -1, self.n_heads, self.dim_per_head
        ).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(
            1, 2
        )
        values = values.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(
            1, 2
        )
        # Calculate the attention
        scores, attn = self.inner_attention(queries, keys, values, attn_mask)
        # Reshape the output
        output = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Apply the linear projection
        output = self.out(output)
        return output, attn
