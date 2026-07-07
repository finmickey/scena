"""Reference conditioning for audio-only inference.

Concatenates encoded reference-audio latents onto the main audio latent
sequence so the transformer can attend to them. In eager PyTorch the
references are concatenated at their true length -- no padding and no
attention mask are needed (unlike the JAX/JIT implementation, which pads to a
static shape and masks the padding).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncodedRefCond:
    """A single encoded reference-audio condition.

    Attributes:
        latents: Patchified VAE-encoded latent tokens, shape ``(B, T_ref, D)``.
        positions: Positional coordinates, shape ``(B, C, T_ref, ...)``.
    """

    latents: torch.Tensor
    positions: torch.Tensor


def prepare_ref_conds_inputs(
    latents: torch.Tensor,
    positions: torch.Tensor,
    timesteps: torch.Tensor,
    ref_conds: dict[int, EncodedRefCond],
    max_ref_conds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append reference-audio latents to the main audio sequence.

    Each present reference ``k`` is concatenated at its true token length; a
    ``ref_cond_ids`` tensor marks main tokens (0) vs reference tokens (k+1).
    References are clean context, so their per-token timestep is 0.

    Args:
        latents: Main audio latents, shape ``(B, N, D)``.
        positions: Main positional coordinates, shape ``(B, C, N, ...)``.
        timesteps: Main per-token timesteps, shape ``(B, N[, 1])``.
        ref_conds: Mapping ref index -> :class:`EncodedRefCond`.
        max_ref_conds: Maximum number of reference slots the model supports.

    Returns:
        ``(latents, positions, timesteps, ref_cond_ids)`` each extended along
        the token dimension by the concatenated reference tokens.
    """
    b, n, _d = latents.shape
    device = latents.device
    dtype = latents.dtype
    ref_cond_ids = torch.zeros(b, n, dtype=torch.long, device=device)  # 0 = main tokens

    for ref_idx in range(max_ref_conds):
        if ref_idx not in ref_conds:
            continue
        ref = ref_conds[ref_idx]
        t_ref = ref.latents.shape[1]

        latents = torch.cat([latents, ref.latents.to(dtype=dtype, device=device)], dim=1)
        positions = torch.cat([positions, ref.positions.to(dtype=positions.dtype, device=device)], dim=2)

        ts_shape = list(timesteps.shape)
        ts_shape[1] = t_ref
        ref_ts = torch.zeros(ts_shape, dtype=timesteps.dtype, device=timesteps.device)
        timesteps = torch.cat([timesteps, ref_ts], dim=1)

        ref_ids = torch.full((b, t_ref), ref_idx + 1, dtype=torch.long, device=device)
        ref_cond_ids = torch.cat([ref_cond_ids, ref_ids], dim=1)

    return latents, positions, timesteps, ref_cond_ids


def slice_ref_conds_output(output: torch.Tensor, original_seq_len: int) -> torch.Tensor:
    """Slice the model output back to the main (non-reference) sequence length."""
    return output[:, :original_seq_len, :]
