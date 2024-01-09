import torch


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def make_source_mask(source_ids, source_pad_id):
    return (source_ids != source_pad_id).unsqueeze(-2)


def make_target_mask(target_ids):
    batch_size, len_target = target_ids.size()
    subsequent_mask = (
        1
        - torch.triu(
            torch.ones((1, len_target, len_target), device=target_ids.device),
            diagonal=1,
        )
    ).bool()
    return subsequent_mask
