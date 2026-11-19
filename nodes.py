"""ComfyUI custom node for Personality Warp (middle-layer potential perturbation)."""

from __future__ import annotations

from .perturbation import build_personality_patches


class PersonalityWarpNode:
    """Apply a deterministic, personality-ID-seeded weight perturbation to middle layers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "personality_id": (
                    "STRING",
                    {
                        "default": "persona_001",
                        "multiline": False,
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "display": "number",
                    },
                ),
                "mode": (["low_rank", "gaussian", "rademacher"],),
                "layer_scope": (["middle_only", "mid_band", "custom_range"],),
                "target_part": (
                    ["attention_and_mlp", "attention_only", "mlp_only", "all_weights"],
                ),
            },
            "optional": {
                "rank": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                    },
                ),
                "layer_start": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "layer_end": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "warp_info")
    FUNCTION = "apply_warp"
    CATEGORY = "model_patches/experimental"

    def apply_warp(
        self,
        model,
        personality_id: str,
        strength: float,
        mode: str,
        layer_scope: str,
        target_part: str,
        rank: int = 4,
        layer_start: float = 0.35,
        layer_end: float = 0.65,
    ):
        cloned = model.clone()

        if strength == 0.0:
            return (cloned, f"PersonalityWarp: bypassed (strength=0.0, id={personality_id})")

        diffusion_model = getattr(cloned.model, "diffusion_model", None)
        if diffusion_model is not None:
            param_items = list(diffusion_model.named_parameters())
            key_prefix = "diffusion_model."
        else:
            state = cloned.model_state_dict()
            param_items = [
                (k, v) for k, v in state.items() if k.startswith("diffusion_model.")
            ]
            key_prefix = ""

        patches = build_personality_patches(
            param_items=param_items,
            personality_id=personality_id,
            strength=strength,
            layer_scope=layer_scope,
            target_part=target_part,
            mode=mode,
            rank=rank,
            layer_start=layer_start,
            layer_end=layer_end,
            key_prefix=key_prefix,
        )

        if patches:
            cloned.add_patches(patches, strength_patch=1.0, strength_model=1.0)

        info = (
            f"PersonalityWarp | id={personality_id} | mode={mode} | "
            f"scope={layer_scope} | part={target_part} | "
            f"strength={strength:.4f} | patched_tensors={len(patches)}"
        )
        return (cloned, info)


NODE_CLASS_MAPPINGS = {
    "PersonalityWarp": PersonalityWarpNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonalityWarp": "Personality Warp (Middle-Layer Perturbation)",
}
