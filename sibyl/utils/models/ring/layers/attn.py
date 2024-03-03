from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from sibyl.utils.models.informer.masking import TriangularCausalMask, ProbMask


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

    def forward(
        self, queries, keys, values, attn_mask
    ) -> tuple[Tensor, Tensor] or Tensor:
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

        A = A if self.output_attention else None

        return V.contiguous(), A


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        """
        Implements Probabilistic Attention mechanism.

        :param mask_flag: If True, masks future positions to enforce causality.
        :param factor: Factor by which to reduce the number of keys.
        :param scale: Scale factor for the attention scores.
        :param attention_dropout: Dropout rate for attention weights.
        :param output_attention: If True, outputs the attention weights.
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top) -> tuple[Tensor, Tensor]:
        """
        Calculates the probabilistic scores between queries and keys.

        :param Q: Queries.
        :param K: Keys.
        :param sample_k: Number of keys to be sampled.
        :param n_top: Number of top elements to be selected.

        :return: The scores and indices of the top elements.
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(
            -2
        )

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q) -> Tensor:
        """
        Initializes the context for the attention mechanism.

        Note: This method calculates the initial context either by taking the mean or
        the cumulative sum of the value matrix, depending on whether masking is applied.

        :param V: The value matrix.
        :param L_Q: The length of the query sequence.

        :return: The initial context tensor.
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(
        self, context_in, V, scores, index, L_Q, attn_mask
    ) -> tuple[Tensor, Tensor] or Tensor:
        """
        Updates the context tensor with the attention mechanism.

        :param context_in: The initial context tensor.
        :param V: The value matrix.
        :param scores: Attention scores.
        :param index: Indices of the top elements.
        :param L_Q: The length of the query sequence.
        :param attn_mask: The attention mask.

        :return: The updated context tensor.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask) -> tuple[Tensor, Tensor]:
        """
        Forward pass for the Probabilistic Attention mechanism.

        :param queries: Queries.
        :param keys: Keys.
        :param values: Values.
        :param attn_mask: Attention mask.

        :returns: A tuple containing the output of the attention mechanism and the attention weights.
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn


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

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass for the AttentionLayer.

        :param queries: Queries.
        :param keys: Keys.
        :param values: Values.
        :param attn_mask: Attention mask.

        :return: A tuple containing the output of the attention mechanism and the attention weights.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
