#!/usr/bin/env python3
"""Create held-out split files for unseen-data forecaster tests.

The split files are intentionally plain text patient lists so they can be used
by both training and evaluation scripts on Cradle.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _safe_name(value: str) -> str:
    value = value.replace("+", "pos").replace("-", "neg")
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _write_list(path: Path, patient_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(patient_ids) + ("\n" if patient_ids else ""))


def _graph_patients(graphs_root: Path) -> set[str]:
    return {path.stem for path in graphs_root.glob("*.pt")}


def _locked_stratified_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = seed
    test_parts = []
    train_parts = []

    strat = df["subtype"].astype(str) + "__pcr" + df["pCR"].astype(str)
    for _, group in df.groupby(strat):
        n_test = max(1, int(round(len(group) * test_frac)))
        n_test = min(n_test, len(group) - 1) if len(group) > 1 else 1
        test = group.sample(n=n_test, random_state=rng)
        train = group.drop(test.index)
        test_parts.append(test)
        train_parts.append(train)
        rng += 1

    test_df = pd.concat(test_parts).sort_values("patient_id")
    train_df = pd.concat(train_parts).sort_values("patient_id")
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=Path("datasets/ispy2/cohort.parquet"))
    parser.add_argument("--patient-list", type=Path, default=Path("reports/all_patients_4visit.txt"))
    parser.add_argument("--graphs-root", type=Path, default=Path("datasets/ispy2/graphs_consistent"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/unseen_forecaster_splits"))
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_pids = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    graph_pids = _graph_patients(args.graphs_root)
    usable = set(all_pids).intersection(graph_pids)

    cohort = pd.read_parquet(args.cohort)
    df = cohort[cohort["patient_id"].isin(usable)].copy()
    df = df.sort_values("patient_id").reset_index(drop=True)

    manifest: list[dict] = []

    # 1. Collection holdout: train on one collection, test on the other.
    for collection in sorted(df["collection"].dropna().unique()):
        test_df = df[df["collection"] == collection]
        train_df = df[df["collection"] != collection]
        name = f"holdout_collection_{_safe_name(str(collection))}"
        train_path = args.out_dir / f"{name}_train.txt"
        test_path = args.out_dir / f"{name}_test.txt"
        _write_list(train_path, train_df["patient_id"].tolist())
        _write_list(test_path, test_df["patient_id"].tolist())
        manifest.append({
            "name": name,
            "kind": "collection_holdout",
            "held_out": str(collection),
            "train_list": str(train_path),
            "test_list": str(test_path),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        })

    # 2. Subtype holdout: train on all other subtypes, test on held-out subtype.
    for subtype in sorted(df["subtype"].dropna().unique()):
        test_df = df[df["subtype"] == subtype]
        train_df = df[df["subtype"] != subtype]
        name = f"holdout_subtype_{_safe_name(str(subtype))}"
        train_path = args.out_dir / f"{name}_train.txt"
        test_path = args.out_dir / f"{name}_test.txt"
        _write_list(train_path, train_df["patient_id"].tolist())
        _write_list(test_path, test_df["patient_id"].tolist())
        manifest.append({
            "name": name,
            "kind": "subtype_holdout",
            "held_out": str(subtype),
            "train_list": str(train_path),
            "test_list": str(test_path),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        })

    # 3. Locked stratified test set from all graph-ready patients.
    train_df, test_df = _locked_stratified_split(df, args.test_frac, args.seed)
    name = "locked_stratified_test"
    train_path = args.out_dir / f"{name}_train.txt"
    test_path = args.out_dir / f"{name}_test.txt"
    _write_list(train_path, train_df["patient_id"].tolist())
    _write_list(test_path, test_df["patient_id"].tolist())
    manifest.append({
        "name": name,
        "kind": "locked_stratified_test",
        "held_out": f"{args.test_frac:.0%}",
        "train_list": str(train_path),
        "test_list": str(test_path),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(manifest)} unseen-data test definitions to {args.out_dir}")
    for item in manifest:
        print(f"{item['name']}: train={item['n_train']} test={item['n_test']}")


if __name__ == "__main__":
    main()
