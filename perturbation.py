"""Deterministic middle-layer weight perturbation for ComfyUI ModelPatcher.

Core idea:
Given a `personality_id`, deterministically generate small weight perturbations
and apply them to the middle layers of a diffusion model via ComfyUI's
non-destructive `ModelPatcher.add_patches` API.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, Iterable, List, Tuple

import torch

# Keywords used to identify parameter roles across UNet (SD1.5/SDXL) and DiT (FLUX/SD3).
ATTENTION_KEYWORDS = (
    "attn",
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "qkv",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)
MLP_KEYWORDS = (
    "mlp",
    "ff.",
    ".ff",
    "linear1",
    "linear2",
    "proj_in",
    "proj_out",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
)
SKIP_KEYWORDS = (
    "bias",
    "norm",
    "time_embed",
    "label_emb",
    "text_embed",
    "vector_in",
    "txt_in",
    "img_in",
    "final_layer",
    "conv_in",
    "conv_out",
    "out.2",
)


def seed_from_key(personality_id: str, param_key: str) -> int:
    """Derive a deterministic 63-bit integer seed from personality_id and param_key."""
    digest = hashlib.sha256(f"{personality_id}::{param_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _extract_block_order_key(param_key: str) -> Tuple[int, int, str]:
    """Order diffusion_model blocks from input -> middle -> output."""
    # UNet (SD1.5 / SDXL)
    m = re.search(r"input_blocks\.(\d+)\.", param_key)
    if m:
        return (0, int(m.group(1)), param_key)
    if "middle_block." in param_key:
        m_mid = re.search(r"middle_block\.(\d+)\.", param_key)
        return (1, int(m_mid.group(1)) if m_mid else 0, param_key)
    m = re.search(r"output_blocks\.(\d+)\.", param_key)
    if m:
        return (2, int(m.group(1)), param_key)

    # DiT / FLUX / SD3
    m = re.search(r"(?:double_blocks|joint_blocks|transformer_blocks|blocks)\.(\d+)\.", param_key)
    if m:
        return (1, int(m.group(1)), param_key)
    m = re.search(r"single_blocks\.(\d+)\.", param_key)
    if m:
        return (2, int(m.group(1)), param_key)

    return (3, 0, param_key)


def _matches_target_part(param_key: str, target_part: str) -> bool:
    lower = param_key.lower()
    if any(skip in lower for skip in SKIP_KEYWORDS):
        return False
    if not lower.endswith(".weight"):
        return False

    if target_part == "all_weights":
        return True
    if target_part == "attention_only":
        return any(k in lower for k in ATTENTION_KEYWORDS)
    if target_part == "mlp_only":
        return any(k in lower for k in MLP_KEYWORDS)
    # Default: "attention_and_mlp"
    return any(k in lower for k in ATTENTION_KEYWORDS) or any(k in lower for k in MLP_KEYWORDS)


def select_target_keys(
    param_items: Iterable[Tuple[str, torch.Tensor]],
    layer_scope: str = "middle_only",
    target_part: str = "attention_and_mlp",
    layer_start: float = 0.35,
    layer_end: float = 0.65,
) -> List[str]:
    """Select parameter keys to perturb based on architecture, scope, and part."""
    candidates: List[Tuple[str, torch.Tensor]] = []
    for key, tensor in param_items:
        if tensor.ndim < 2:
            continue
        if _matches_target_part(key, target_part):
            candidates.append((key, tensor))

    if not candidates:
        return []

    keys = [k for k, _ in candidates]
    has_unet_middle = any("middle_block." in k for k in keys)

    if layer_scope == "middle_only" and has_unet_middle:
        return [k for k in keys if "middle_block." in k]

    if layer_scope == "mid_band" and has_unet_middle:
        # Include deep input_blocks (6+), middle_block, and early output_blocks (0..5)
        selected = []
        for k in keys:
            if "middle_block." in k:
                selected.append(k)
                continue
            m_in = re.search(r"input_blocks\.(\d+)\.", k)
            if m_in and int(m_in.group(1)) >= 6:
                selected.append(k)
                continue
            m_out = re.search(r"output_blocks\.(\d+)\.", k)
            if m_out and int(m_out.group(1)) <= 5:
                selected.append(k)
                continue
        if selected:
            return selected

    # Generic / DiT / custom_range: sort blocks in forward order and slice by [start, end]
    ordered = sorted(keys, key=_extract_block_order_key)
    if layer_scope == "middle_only":
        start_frac, end_frac = 0.35, 0.65
    elif layer_scope == "mid_band":
        start_frac, end_frac = 0.25, 0.75
    else:
        start_frac = max(0.0, min(1.0, layer_start))
        end_frac = max(start_frac, min(1.0, layer_end))

    n = len(ordered)
    lo = int(math.floor(n * start_frac))
    hi = max(lo + 1, int(math.ceil(n * end_frac)))
    return ordered[lo:hi]


def generate_perturbation(
    weight: torch.Tensor,
    personality_id: str,
    param_key: str,
    strength: float,
    mode: str = "low_rank",
    rank: int = 4,
) -> torch.Tensor:
    """Generate a deterministic perturbation tensor matching `weight`'s shape and dtype."""
    if strength == 0.0:
        return torch.zeros_like(weight, device="cpu")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed_from_key(personality_id, param_key))

    w_float = weight.detach().to(device="cpu", dtype=torch.float32)
    w_std = float(w_float.std().item())
    if w_std == 0.0:
        w_std = 1e-4

    shape = w_float.shape
    if mode == "low_rank" and w_float.ndim >= 2:
        out_dim = shape[0]
        in_dim = int(math.prod(shape[1:]))
        eff_rank = max(1, min(rank, out_dim, in_dim))
        a = torch.randn((out_dim, eff_rank), generator=gen, dtype=torch.float32)
        b = torch.randn((in_dim, eff_rank), generator=gen, dtype=torch.float32)
        raw = (a @ b.T) / math.sqrt(eff_rank)
        raw = raw.reshape(shape)
    elif mode == "rademacher":
        raw = torch.randint(0, 2, shape, generator=gen, dtype=torch.float32) * 2.0 - 1.0
    else:
        # "gaussian"
        raw = torch.randn(shape, generator=gen, dtype=torch.float32)

    raw_std = float(raw.std().item())
    if raw_std > 1e-8:
        raw = raw / raw_std

    delta = raw * (strength * w_std)
    return delta.to(dtype=weight.dtype)


def build_personality_patches(
    param_items: Iterable[Tuple[str, torch.Tensor]],
    personality_id: str,
    strength: float,
    layer_scope: str = "middle_only",
    target_part: str = "attention_and_mlp",
    mode: str = "low_rank",
    rank: int = 4,
    layer_start: float = 0.35,
    layer_end: float = 0.65,
    key_prefix: str = "diffusion_model.",
) -> Dict[str, Tuple[torch.Tensor]]:
    """Build a ComfyUI ModelPatcher-compatible patches dictionary."""
    if strength == 0.0:
        return {}

    item_map = {k: v for k, v in param_items}
    selected_keys = select_target_keys(
        item_map.items(),
        layer_scope=layer_scope,
        target_part=target_part,
        layer_start=layer_start,
        layer_end=layer_end,
    )

    patches: Dict[str, Tuple[torch.Tensor]] = {}
    for key in selected_keys:
        full_key = key if key.startswith(key_prefix) else f"{key_prefix}{key}"
        delta = generate_perturbation(
            weight=item_map[key],
            personality_id=personality_id,
            param_key=key,
            strength=strength,
            mode=mode,
            rank=rank,
        )
        patches[full_key] = (delta,)

    return patches
