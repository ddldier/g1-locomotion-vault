#!/usr/bin/env python3
"""Offline sanity check for the G1 Speed Vault NPZ and VaultBox placement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--plot", type=Path, default=None, help="Optional top/side view PNG")
    args = parser.parse_args()

    data = np.load(args.npz)
    rate = float(np.asarray(data["framerate"]).reshape(-1)[0])
    base = np.asarray(data["base_pos_w"], dtype=np.float64)
    size = np.asarray(data["obstacle_size"], dtype=np.float64)
    obstacle = np.asarray(data["obstacle_pos"], dtype=np.float64)

    lo = obstacle - size / 2.0
    hi = obstacle + size / 2.0
    inside_x = (base[:, 0] >= lo[0]) & (base[:, 0] <= hi[0])
    inside_y = (base[:, 1] >= lo[1]) & (base[:, 1] <= hi[1])
    crossing = np.flatnonzero(inside_x & inside_y)

    print(f"frames={len(base)} framerate={rate:g} duration={(len(base)-1)/rate:.3f}s")
    print(f"base x=[{base[:, 0].min():.3f}, {base[:, 0].max():.3f}] "
          f"y=[{base[:, 1].min():.3f}, {base[:, 1].max():.3f}] "
          f"z=[{base[:, 2].min():.3f}, {base[:, 2].max():.3f}]")
    print(f"obstacle center={obstacle.tolist()} size={size.tolist()}")
    print(f"obstacle AABB x=[{lo[0]:.3f}, {hi[0]:.3f}] "
          f"y=[{lo[1]:.3f}, {hi[1]:.3f}] z=[{lo[2]:.3f}, {hi[2]:.3f}]")
    if crossing.size:
        print(f"base enters obstacle AABB at frame {crossing[0]} ({crossing[0]/rate:.3f}s) "
              f"and leaves at frame {crossing[-1]} ({crossing[-1]/rate:.3f}s)")
        print(f"crossing base z range={base[crossing, 2].min():.3f}..{base[crossing, 2].max():.3f}")
    else:
        print("WARNING: base never enters the obstacle horizontal AABB")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 4))
        ax_top.plot(base[:, 0], base[:, 1], label="base trajectory")
        ax_top.add_patch(plt.Rectangle((lo[0], lo[1]), size[0], size[1], fill=False, color="red"))
        ax_top.set_xlabel("x"); ax_top.set_ylabel("y"); ax_top.set_title("top view")
        ax_top.axis("equal"); ax_top.legend()
        t = np.arange(len(base)) / rate
        ax_side.plot(t, base[:, 0], label="base x")
        ax_side.axhspan(lo[0], hi[0], color="red", alpha=0.15, label="box x range")
        ax_side.set_xlabel("time (s)"); ax_side.set_ylabel("x"); ax_side.legend()
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
