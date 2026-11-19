"""Unit tests for Personality Warp perturbation and ComfyUI node wrapper."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_package():
    spec = importlib.util.spec_from_file_location(
        "comfyui_personality_warp",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_personality_warp"] = pkg
    assert spec.loader is not None
    spec.loader.exec_module(pkg)
    return pkg


_load_package()
from comfyui_personality_warp import NODE_CLASS_MAPPINGS  # noqa: E402
from comfyui_personality_warp.perturbation import (  # noqa: E402
    generate_perturbation,
    select_target_keys,
)


def _dummy_unet_params():
    return [
        ("input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight", torch.randn(32, 32)),
        ("input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight", torch.randn(32, 32)),
        ("middle_block.1.transformer_blocks.0.attn1.to_q.weight", torch.randn(64, 64)),
        ("middle_block.1.transformer_blocks.0.ff.net.0.proj.weight", torch.randn(128, 64)),
        ("middle_block.1.norm.weight", torch.randn(64)),
        ("middle_block.1.transformer_blocks.0.attn1.to_q.bias", torch.randn(64)),
        ("output_blocks.2.1.transformer_blocks.0.attn1.to_v.weight", torch.randn(32, 32)),
        ("output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight", torch.randn(32, 32)),
        ("time_embed.0.weight", torch.randn(64, 32)),
    ]


def _dummy_dit_params():
    params = []
    for i in range(10):
        params.append((f"double_blocks.{i}.img_attn.qkv.weight", torch.randn(64, 32)))
        params.append((f"double_blocks.{i}.img_mlp.0.weight", torch.randn(64, 32)))
        params.append((f"double_blocks.{i}.img_norm1.weight", torch.randn(32)))
    return params


class _MockModelPatcher:
    def __init__(self, diffusion_model):
        self.model = SimpleNamespace(diffusion_model=diffusion_model)
        self.patches = {}

    def clone(self):
        c = _MockModelPatcher(self.model.diffusion_model)
        c.patches = dict(self.patches)
        return c

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        for k, v in patches.items():
            self.patches[k] = (strength_patch, v, strength_model)
        return list(patches.keys())


class TestPersonalityWarp(unittest.TestCase):
    def test_unet_middle_only_selection(self):
        keys = select_target_keys(_dummy_unet_params(), layer_scope="middle_only")
        self.assertEqual(
            keys,
            [
                "middle_block.1.transformer_blocks.0.attn1.to_q.weight",
                "middle_block.1.transformer_blocks.0.ff.net.0.proj.weight",
            ],
        )

    def test_unet_mid_band_selection(self):
        keys = select_target_keys(_dummy_unet_params(), layer_scope="mid_band")
        self.assertIn("input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight", keys)
        self.assertIn("middle_block.1.transformer_blocks.0.attn1.to_q.weight", keys)
        self.assertIn("output_blocks.2.1.transformer_blocks.0.attn1.to_v.weight", keys)
        self.assertNotIn("input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight", keys)
        self.assertNotIn("output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight", keys)

    def test_dit_middle_selection(self):
        keys = select_target_keys(_dummy_dit_params(), layer_scope="middle_only")
        self.assertGreater(len(keys), 0)
        self.assertTrue(all("double_blocks." in k for k in keys))
        self.assertFalse(any("double_blocks.0." in k for k in keys))
        self.assertFalse(any("double_blocks.9." in k for k in keys))

    def test_perturbation_determinism_and_scale(self):
        w = torch.randn(64, 64, dtype=torch.float16)
        for mode in ("low_rank", "gaussian", "rademacher"):
            with self.subTest(mode=mode):
                d1 = generate_perturbation(
                    w, "alice", "middle_block.1.to_q.weight", strength=0.05, mode=mode
                )
                d2 = generate_perturbation(
                    w, "alice", "middle_block.1.to_q.weight", strength=0.05, mode=mode
                )
                d_other = generate_perturbation(
                    w, "bob", "middle_block.1.to_q.weight", strength=0.05, mode=mode
                )

                self.assertEqual(d1.shape, w.shape)
                self.assertEqual(d1.dtype, w.dtype)
                self.assertTrue(torch.equal(d1, d2))
                self.assertFalse(torch.equal(d1, d_other))

                ratio = d1.float().std().item() / w.float().std().item()
                self.assertGreaterEqual(ratio, 0.03)
                self.assertLessEqual(ratio, 0.07)

    def test_node_non_destructive_patching(self):
        class DummyDiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.middle_block = torch.nn.ModuleDict(
                    {"attn_to_q": torch.nn.Linear(32, 32, bias=False)}
                )

        dm = DummyDiffusionModel()
        base_patcher = _MockModelPatcher(dm)

        node = NODE_CLASS_MAPPINGS["PersonalityWarp"]()
        warped_a, info_a = node.apply_warp(
            model=base_patcher,
            personality_id="persona_A",
            strength=0.02,
            mode="low_rank",
            layer_scope="middle_only",
            target_part="attention_and_mlp",
        )
        warped_b, _ = node.apply_warp(
            model=base_patcher,
            personality_id="persona_B",
            strength=0.02,
            mode="low_rank",
            layer_scope="middle_only",
            target_part="attention_and_mlp",
        )

        self.assertEqual(len(base_patcher.patches), 0)
        self.assertIn("diffusion_model.middle_block.attn_to_q.weight", warped_a.patches)
        self.assertIn("patched_tensors=1", info_a)
        delta_a = warped_a.patches["diffusion_model.middle_block.attn_to_q.weight"][1][0]
        delta_b = warped_b.patches["diffusion_model.middle_block.attn_to_q.weight"][1][0]
        self.assertFalse(torch.equal(delta_a, delta_b))


if __name__ == "__main__":
    unittest.main()
