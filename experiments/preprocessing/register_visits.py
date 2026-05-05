#!/usr/bin/env python3
"""Deformable registration pipeline (ANTs SyN) for the digital-twin prototype.

For each patient and each follow-up visit T_k (k = 1, 2, 3):
  1. Load the PE volume from `derived_v2/{pid}/T0` and `derived_v2/{pid}/Tk`
  2. Wrap as ANTs images using origin/spacing from meta.json
  3. Register T_k -> T0 with SyN (rigid + affine + diffeomorphic)
  4. Save:
       - registered PE volume   (Tk -> T0 space)
       - forward warp           (point T0 -> Tk, used to transport T0 supervoxel mask)
       - inverse warp           (point Tk -> T0)
       - registered tumour mask (binary, transported)
       - registration metrics   (final cross-correlation, run time)

Outputs land in `datasets/ispy2/registered/{pid}/Tk_to_T0/`.

Usage (single patient sanity check):
    python experiments/preprocessing/register_visits.py \
        --patient ISPY2-559021 --derived datasets/ispy2/derived_v2

Usage (batch from prototype patient list):
    python experiments/preprocessing/register_visits.py \
        --patient-list reports/prototype_patients.txt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_visit_pe(visit_dir: Path):
    """Return (PE volume as float32 ndarray, origin, spacing, mask_optional)."""
    npz = np.load(visit_dir / "pe_ser.npz")
    pe = npz["pe"].astype(np.float32)

    meta = json.loads((visit_dir / "meta.json").read_text())
    spacing = tuple(float(s) for s in meta["voxel_spacing_mm"])
    origin  = tuple(float(o) for o in meta["origin_mm"])

    # Tumour mask = supervoxel labels > 0 (so registration can be tumour-focused)
    sv = np.load(visit_dir / "supervoxel_labels.npz")["labels"]
    mask = (sv > 0).astype(np.uint8)

    return pe, origin, spacing, mask, sv


def to_ants(volume: np.ndarray, origin, spacing):
    """Wrap a numpy volume into an ANTs image with proper physical metadata.

    The volumes are stored (z, y, x). ANTs uses (x, y, z) ordering, so we
    transpose. Origin and spacing are likewise reordered.
    """
    import ants
    arr = np.ascontiguousarray(volume.transpose(2, 1, 0))
    img = ants.from_numpy(
        arr,
        origin=(origin[2], origin[1], origin[0]),
        spacing=(spacing[2], spacing[1], spacing[0]),
        direction=np.eye(3),
    )
    return img


def from_ants(img):
    """Convert ANTs image back to (z, y, x) numpy."""
    arr = img.numpy()  # (x, y, z)
    return np.ascontiguousarray(arr.transpose(2, 1, 0))


def register_visit_to_t0(t0_dir: Path, tk_dir: Path, out_dir: Path,
                         use_mask: bool = True, save_arrays: bool = True):
    """Register T_k -> T0 with rigid+affine+SyN. Save warp fields and outputs."""
    import ants
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_pe, t0_origin, t0_spacing, t0_mask, t0_sv = load_visit_pe(t0_dir)
    tk_pe, tk_origin, tk_spacing, tk_mask, _    = load_visit_pe(tk_dir)

    # Normalise PE volumes (clip percentiles, scale 0..1) so registration is
    # robust to inter-visit intensity drift caused by treatment response.
    def _norm(vol):
        lo, hi = np.percentile(vol, [1, 99])
        v = np.clip(vol, lo, hi)
        return (v - lo) / max(hi - lo, 1e-6)

    fixed  = to_ants(_norm(t0_pe), t0_origin, t0_spacing)   # T0 = fixed
    moving = to_ants(_norm(tk_pe), tk_origin, tk_spacing)   # Tk = moving

    fixed_mask = None
    if use_mask:
        # Slightly dilate the mask so registration sees a small ring of
        # surrounding parenchyma — pure tumour mask can over-constrain.
        from scipy.ndimage import binary_dilation
        dil = binary_dilation(t0_mask, iterations=5).astype(np.uint8)
        fixed_mask = to_ants(dil, t0_origin, t0_spacing)

    t_start = time.time()
    reg = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform="SyN",
        mask=fixed_mask,
        verbose=False,
    )
    elapsed = time.time() - t_start

    # Final cross-correlation between fixed and warped-moving (in mask if available)
    warped = reg["warpedmovout"]
    cc = float(ants.image_similarity(fixed, warped, metric_type="MeanSquares") * -1)

    if save_arrays:
        np.savez_compressed(
            out_dir / "registered.npz",
            warped_pe=from_ants(warped).astype(np.float32),
            fixed_pe=t0_pe,
            moving_pe=tk_pe,
            t0_mask=t0_mask, tk_mask=tk_mask, t0_sv=t0_sv,
            t0_origin=np.array(t0_origin, dtype=np.float32),
            t0_spacing=np.array(t0_spacing, dtype=np.float32),
        )

    # Persist transform files (so we can reapply later for masks/labels).
    # ANTs returns .nii.gz (warp fields) and .mat (affine) — copy verbatim
    # so the file extensions ITK expects are preserved.
    import shutil
    saved_fwd = []
    for i, f in enumerate(reg["fwdtransforms"]):
        ext = "".join(Path(f).suffixes) or ".bin"
        dst = out_dir / f"fwd_{i:02d}{ext}"
        shutil.copy2(f, dst)
        saved_fwd.append(str(dst))
    saved_inv = []
    for i, f in enumerate(reg["invtransforms"]):
        ext = "".join(Path(f).suffixes) or ".bin"
        dst = out_dir / f"inv_{i:02d}{ext}"
        shutil.copy2(f, dst)
        saved_inv.append(str(dst))

    summary = dict(
        elapsed_sec=elapsed,
        cc_negmse=cc,
        fwdtransforms=saved_fwd,
        invtransforms=saved_inv,
        used_mask=bool(use_mask),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def __copy(src, dst):
    import shutil; shutil.copy2(src, dst)


def transport_t0_supervoxels_to_tk(t0_dir: Path, tk_dir: Path,
                                   reg_dir: Path, out_path: Path):
    """Take the T0 supervoxel label volume and transport it into T_k native
    space using the **inverse** transform from the T_k -> T0 registration.
    """
    import ants
    t0_sv = np.load(t0_dir / "supervoxel_labels.npz")["labels"]
    t0_meta = json.loads((t0_dir / "meta.json").read_text())
    tk_meta = json.loads((tk_dir / "meta.json").read_text())
    tk_pe = np.load(tk_dir / "pe_ser.npz")["pe"]
    tk_origin = tuple(float(o) for o in tk_meta["origin_mm"])
    tk_spacing = tuple(float(s) for s in tk_meta["voxel_spacing_mm"])
    t0_origin = tuple(float(o) for o in t0_meta["origin_mm"])
    t0_spacing = tuple(float(s) for s in t0_meta["voxel_spacing_mm"])

    sv_img = to_ants(t0_sv.astype(np.float32), t0_origin, t0_spacing)
    ref_img = to_ants(tk_pe.astype(np.float32), tk_origin, tk_spacing)

    inv = sorted(reg_dir.glob("inv_*.*"))
    inv = [str(p) for p in inv]
    if not inv:
        raise FileNotFoundError(f"No inv transforms in {reg_dir}")

    transported = ants.apply_transforms(
        fixed=ref_img, moving=sv_img,
        transformlist=inv, interpolator="genericLabel",
    )
    arr = from_ants(transported).astype(np.int32)
    np.savez_compressed(
        out_path,
        labels=arr,
        origin=np.array(tk_origin, dtype=np.float32),
        spacing=np.array(tk_spacing, dtype=np.float32),
    )
    return arr


def process_patient(pid: str, derived: Path, out_root: Path,
                    visits=("T1", "T2", "T3")) -> dict:
    t0_dir = derived / pid / "T0"
    if not t0_dir.exists():
        return {"pid": pid, "skipped": True, "reason": "no T0"}

    res = {"pid": pid, "registrations": {}, "transports": {}}
    for v in visits:
        tk_dir = derived / pid / v
        if not tk_dir.exists():
            res["registrations"][v] = {"skipped": True, "reason": "missing visit"}
            continue
        out_dir = out_root / pid / f"{v}_to_T0"
        try:
            summ = register_visit_to_t0(t0_dir, tk_dir, out_dir)
            res["registrations"][v] = summ
            transport_path = out_root / pid / f"{v}_t0sv_in_{v}_space.npz"
            transport_t0_supervoxels_to_tk(t0_dir, tk_dir, out_dir, transport_path)
            res["transports"][v] = str(transport_path)
        except Exception as e:
            res["registrations"][v] = {"error": f"{type(e).__name__}: {e}"}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", help="Process a single patient ID (sanity check)")
    ap.add_argument("--patient-list", type=Path,
                    help="File with one patient ID per line (overrides --patient)")
    ap.add_argument("--derived", type=Path,
                    default=Path("datasets/ispy2/derived_v2"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("datasets/ispy2/registered"))
    args = ap.parse_args()

    if args.patient_list:
        pids = [ln.strip() for ln in args.patient_list.read_text().splitlines() if ln.strip()]
    elif args.patient:
        pids = [args.patient]
    else:
        ap.error("Provide --patient or --patient-list")

    print(f"Processing {len(pids)} patient(s) -> {args.out_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path = args.out_root / "_log.jsonl"
    with log_path.open("a") as fh:
        for i, pid in enumerate(pids):
            print(f"\n[{i+1}/{len(pids)}] {pid}", flush=True)
            res = process_patient(pid, args.derived, args.out_root)
            for v, r in res.get("registrations", {}).items():
                if "elapsed_sec" in r:
                    print(f"  {v}: {r['elapsed_sec']:.1f}s  cc={r['cc_negmse']:.4f}")
                elif "error" in r:
                    print(f"  {v}: ERROR {r['error']}")
                else:
                    print(f"  {v}: skipped ({r.get('reason')})")
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(f"\nLog: {log_path}")


if __name__ == "__main__":
    main()
