#!/usr/bin/env python3
"""Evaluate a trained consistent-graph forecaster on an arbitrary patient list.

This is intended for held-out generalization tests where the training script's
validation split is not the final evaluation set, e.g. train on ISPY2 and test
on ACRIN, or train on all-but-one molecular subtype and test on the held-out
subtype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lsgc.forecaster import LSGCForecaster
from experiments.stage1_forecaster.train_consistent_forecaster import (
    ConsistentSample,
    forward_patient,
    load_sample,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_patient_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _load_samples(graphs_root: Path, patient_ids: Iterable[str]) -> list[ConsistentSample]:
    samples: list[ConsistentSample] = []
    for pid in patient_ids:
        sample = load_sample(graphs_root / f"{pid}.pt")
        if sample is not None:
            samples.append(sample)
    return samples


def _transition_deltas(sample: ConsistentSample) -> list[torch.Tensor]:
    """Return destination-alive position deltas for each transition."""
    deltas: list[torch.Tensor] = []
    off = sample.visit_offsets
    for visit_idx in range(len(off) - 2):
        src = slice(int(off[visit_idx]), int(off[visit_idx + 1]))
        dst = slice(int(off[visit_idx + 1]), int(off[visit_idx + 2]))
        mask = sample.alive[dst].bool()
        if mask.sum() == 0:
            continue
        deltas.append((sample.pos[dst] - sample.pos[src])[mask])
    return deltas


def _mean_delta_by_transition(samples: list[ConsistentSample]) -> list[torch.Tensor]:
    """Compute train-set mean displacement vectors for T0->T1, T1->T2, T2->T3."""
    per_transition: list[list[torch.Tensor]] = [[], [], []]
    for sample in samples:
        for idx, delta in enumerate(_transition_deltas(sample)):
            per_transition[idx].append(delta)

    means: list[torch.Tensor] = []
    for chunks in per_transition:
        if chunks:
            means.append(torch.cat(chunks, dim=0).mean(dim=0))
        else:
            means.append(torch.zeros(3))
    return means


def _baseline_mae(samples: list[ConsistentSample], mean_deltas: list[torch.Tensor] | None = None) -> dict:
    zero_mae: list[float] = []
    mean_mae: list[float] = []
    n_alive = 0

    for sample in samples:
        for idx, delta in enumerate(_transition_deltas(sample)):
            zero_mae.append(delta.norm(dim=-1).mean().item())
            n_alive += int(delta.shape[0])
            if mean_deltas is not None:
                pred = mean_deltas[idx].to(delta)
                mean_mae.append((pred - delta).norm(dim=-1).mean().item())

    out = {
        "zero_mae_mm": float(np.mean(zero_mae)) if zero_mae else float("nan"),
        "n_eval_transitions": int(len(zero_mae)),
        "n_alive_supervoxel_transitions": int(n_alive),
    }
    if mean_deltas is not None:
        out["train_mean_delta_mae_mm"] = (
            float(np.mean(mean_mae)) if mean_mae else float("nan")
        )
    return out


def evaluate(
    checkpoint: Path,
    graphs_root: Path,
    test_list: Path,
    out_json: Path,
    train_list: Path | None = None,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    mean = ckpt["mean"].to(DEVICE)
    std = ckpt["std"].to(DEVICE)

    in_channels = int(ckpt["state_dict"]["embed.weight"].shape[1])
    feat_out_dim = in_channels
    hidden = int(cfg.get("hidden", ckpt["state_dict"]["embed.weight"].shape[0]))
    num_layers = int(cfg.get("num_layers", 2))

    model = LSGCForecaster(
        in_channels=in_channels,
        hidden=hidden,
        num_layers=num_layers,
        feat_out_dim=feat_out_dim,
        use_delta_t=True,
        use_edge_gating=True,
        edge_attr_dim=0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_ids = _read_patient_list(test_list)
    test_samples = _load_samples(graphs_root, test_ids)

    train_samples: list[ConsistentSample] = []
    if train_list is not None:
        train_samples = _load_samples(graphs_root, _read_patient_list(train_list))
    mean_deltas = _mean_delta_by_transition(train_samples) if train_samples else None

    losses: list[float] = []
    mae: list[float] = []
    with torch.no_grad():
        for sample in test_samples:
            loss, info = forward_patient(model, sample, mean, std, DEVICE)
            losses.append(loss.item())
            for transition in info["transitions"]:
                if "pos_mae_mm" in transition:
                    mae.append(transition["pos_mae_mm"])

    metrics = {
        "checkpoint": str(checkpoint),
        "graphs_root": str(graphs_root),
        "test_list": str(test_list),
        "train_list": str(train_list) if train_list is not None else None,
        "n_requested": len(test_ids),
        "n_loaded": len(test_samples),
        "test_loss": float(np.mean(losses)) if losses else float("nan"),
        "test_mae_mm": float(np.mean(mae)) if mae else float("nan"),
        "test_mae_std_over_transitions": float(np.std(mae)) if mae else float("nan"),
    }
    metrics.update(_baseline_mae(test_samples, mean_deltas))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graphs-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        graphs_root=args.graphs_root,
        test_list=args.test_list,
        train_list=args.train_list,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
