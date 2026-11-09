#!/usr/bin/env python3
"""Export a compatible Unified Transition PPO actor and normalizer to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


DEFAULT_CHECKPOINT = Path(
    "/root/gpufree-data/InstinctLab/logs/instinct_rl/g1_unified_transition_ppo/"
    "20261109_182619_g1_unified_transition_ppo_unified_vault_yaw_mix_from11506_"
    "from20261109_174214/model_13507.pt"
)


class ActorInference(nn.Module):
    def __init__(self, checkpoint: dict):
        super().__init__()
        model = checkpoint["model_state_dict"]
        normalizer = checkpoint["policy_normalizer_state_dict"]
        self.register_buffer("mean", normalizer["_mean"].detach().clone())
        self.register_buffer("std", normalizer["_std"].detach().clone())
        self.eps = 1.0e-2
        self.layers = nn.Sequential(
            nn.Linear(2049, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        with torch.no_grad():
            for index, key in zip((0, 2, 4, 6), ("0", "2", "4", "6")):
                self.layers[index].weight.copy_(model[f"actor.{key}.weight"])
                self.layers[index].bias.copy_(model[f"actor.{key}.bias"])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.mean) / (self.std + self.eps)
        return self.layers(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/model_13507_actor.onnx",
    )
    parser.add_argument("--verify-batches", type=int, default=8)
    args = parser.parse_args()

    if args.verify_batches < 1:
        parser.error("--verify-batches must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    iteration = checkpoint.get("iter")
    model = checkpoint.get("model_state_dict", {})
    normalizer = checkpoint.get("policy_normalizer_state_dict", {})
    expected_shapes = {
        "actor.0.weight": (512, 2049),
        "actor.2.weight": (256, 512),
        "actor.4.weight": (128, 256),
        "actor.6.weight": (29, 128),
    }
    for name, expected in expected_shapes.items():
        tensor = model.get(name)
        if tensor is None or tuple(tensor.shape) != expected:
            raise ValueError(
                f"Checkpoint is not compatible: {name} expected {expected}, "
                f"got {None if tensor is None else tuple(tensor.shape)}"
            )
    mean = normalizer.get("_mean")
    if mean is None or tuple(mean.shape) != (1, 2049):
        raise ValueError(
            "Checkpoint policy normalizer must have shape (1, 2049), "
            f"got {None if mean is None else tuple(mean.shape)}"
        )

    actor = ActorInference(checkpoint).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 2049, dtype=torch.float32)
    torch.onnx.export(
        actor,
        sample,
        args.output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.output))

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1502)
    max_error = 0.0
    for _ in range(args.verify_batches):
        observations = rng.normal(size=(4, 2049)).astype(np.float32)
        with torch.inference_mode():
            expected = actor(torch.from_numpy(observations)).numpy()
        actual = session.run(["action"], {"observation": observations})[0]
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    if max_error > 2.0e-5:
        raise RuntimeError(f"ONNX parity check failed: max abs error {max_error:.3g}")

    print(f"Checkpoint: {args.checkpoint} (iteration {iteration})")
    print(f"ONNX:       {args.output}")
    print("Input:      [batch, 2049]")
    print("Output:     [batch, 29]")
    print(f"Parity:     passed, max abs error {max_error:.3g}")


if __name__ == "__main__":
    main()
