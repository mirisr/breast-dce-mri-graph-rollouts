#!/usr/bin/env python3
"""Create T-style visit aliases for Breast-MRI-NACT-Pilot derived bundles."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, *, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
        return
    try:
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def visit_map(visit_count: int) -> dict[str, str]:
    if visit_count == 4:
        return {"V1": "T0", "V2": "T1", "V3": "T2", "V4": "T3"}
    if visit_count == 3:
        return {"V1": "T0", "V2": "T1", "V3": "T3"}
    raise ValueError("--visit-count must be 3 or 4")


def make_patient_aliases(pid: str, source_root: Path, alias_root: Path, mapping: dict[str, str], *, copy: bool) -> bool:
    src_patient = source_root / pid
    if not src_patient.exists():
        return False
    for src_visit, dst_visit in mapping.items():
        src = src_patient / src_visit
        if not src.exists():
            return False
        link_or_copy(src, alias_root / pid / dst_visit, copy=copy)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/derived_v2"))
    parser.add_argument("--alias-root", type=Path, default=Path("datasets/breast_mri_nact_pilot/derived_4visit"))
    parser.add_argument("--patient-list", type=Path, required=True)
    parser.add_argument("--out-patient-list", type=Path, required=True)
    parser.add_argument("--visit-count", type=int, choices=(3, 4), default=4)
    parser.add_argument("--copy", action="store_true", help="Copy directories instead of symlinking.")
    args = parser.parse_args()

    patients = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    mapping = visit_map(args.visit_count)
    ok: list[str] = []
    failed: list[str] = []
    for pid in patients:
        try:
            if make_patient_aliases(pid, args.source_root, args.alias_root, mapping, copy=args.copy):
                ok.append(pid)
            else:
                failed.append(pid)
        except Exception as exc:
            failed.append(f"{pid}: {type(exc).__name__}: {exc}")

    args.out_patient_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_patient_list.write_text("\n".join(ok) + ("\n" if ok else ""), encoding="utf-8")
    print(
        f"[nact-visit-aliases] visit_count={args.visit_count} ok={len(ok)} "
        f"failed={len(failed)} alias_root={args.alias_root}"
    )
    if failed:
        print("\n".join(failed))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
