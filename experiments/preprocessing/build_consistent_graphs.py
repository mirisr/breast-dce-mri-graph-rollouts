#!/usr/bin/env python3
"""Build 'consistent graph' representation for the digital-twin prototype.

Unlike `graphs_v3_full` (where every visit has its own re-segmented supervoxels
and we hope NN matching can stitch them together), the consistent graph uses
**persistent node identities** across visits. The procedure:

1. Take the T0 supervoxel partition as canonical (N supervoxels, ids 1..N).
2. For each follow-up visit T_k, use the registration result to transport the
   T0 label volume into T_k native space (file:
   `datasets/ispy2/registered/{pid}/T{k}_t0sv_in_T{k}_space.npz`).
3. For every (visit, supervoxel id) pair, compute features from the voxels in
   T_k native space that carry that label:
       - centroid (mm, in T_k physical space)
       - voxel count (volume_ml)
       - mean PE, std PE
       - mean SER
       - alive flag (1 if voxel_count > 0)
4. Stack 4 visits into a single graph object. Spatial edges are per-visit kNN
   on the centroids; temporal edges are **deterministic** node_i_v → node_i_v+1.

Output: `datasets/ispy2/graphs_consistent/{pid}.pt`

Run:
    python experiments/preprocessing/build_consistent_graphs.py \
        --patient-list reports/prototype_patients.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


VISITS = ("T0", "T1", "T2", "T3")
FEATURE_NAMES = [
    "voxel_count_log",   # log1p(voxel_count)
    "volume_ml",         # voxel_count * spacing_product / 1000
    "pe_mean", "pe_std",
    "ser_mean", "ser_std",
]


def per_label_stats(labels_vol: np.ndarray, pe_vol: np.ndarray, ser_vol: np.ndarray,
                    spacing_zyx, origin_zyx, n_supervoxels: int) -> dict:
    """For each supervoxel id 1..n_supervoxels, compute centroid + intensity stats.

    Returns dict with keys:
        centroids (N, 3)  in physical mm
        voxel_count (N,)
        pe_mean, pe_std (N,)
        ser_mean, ser_std (N,)
        alive (N,) bool
    """
    N = n_supervoxels
    centroids = np.zeros((N, 3), dtype=np.float32)
    voxel_count = np.zeros(N, dtype=np.int64)
    pe_mean = np.zeros(N, dtype=np.float32)
    pe_std = np.zeros(N, dtype=np.float32)
    ser_mean = np.zeros(N, dtype=np.float32)
    ser_std = np.zeros(N, dtype=np.float32)

    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    origin  = np.asarray(origin_zyx,  dtype=np.float32)

    Z, Y, X = labels_vol.shape
    flat = labels_vol.reshape(-1)

    # Linear voxel-to-mm: idx (z,y,x) -> origin + idx * spacing  (mm)
    # Build (z,y,x) coordinates lazily via np.unravel_index per id (or fully)
    nz_idx = np.flatnonzero(flat > 0)
    if nz_idx.size == 0:
        return dict(centroids=centroids, voxel_count=voxel_count,
                    pe_mean=pe_mean, pe_std=pe_std,
                    ser_mean=ser_mean, ser_std=ser_std,
                    alive=np.zeros(N, dtype=bool))

    z_idx, y_idx, x_idx = np.unravel_index(nz_idx, (Z, Y, X))
    ids = flat[nz_idx]
    pe_vals = pe_vol[z_idx, y_idx, x_idx]
    ser_vals = ser_vol[z_idx, y_idx, x_idx]

    z_mm = origin[0] + z_idx * spacing[0]
    y_mm = origin[1] + y_idx * spacing[1]
    x_mm = origin[2] + x_idx * spacing[2]

    for lid in np.unique(ids):
        if lid <= 0 or lid > N:
            continue
        sel = ids == lid
        n_vox = sel.sum()
        if n_vox == 0:
            continue
        i = lid - 1
        centroids[i, 0] = z_mm[sel].mean()
        centroids[i, 1] = y_mm[sel].mean()
        centroids[i, 2] = x_mm[sel].mean()
        voxel_count[i] = n_vox
        pe_mean[i] = pe_vals[sel].mean()
        pe_std[i]  = pe_vals[sel].std()
        ser_mean[i] = ser_vals[sel].mean()
        ser_std[i]  = ser_vals[sel].std()

    alive = voxel_count > 0
    # For dropped supervoxels (alive == False) at later visits, use the *T0*
    # centroid as a placeholder so position vectors are never NaN. Alive flag
    # tells the model which nodes actually exist at that visit.
    return dict(centroids=centroids, voxel_count=voxel_count,
                pe_mean=pe_mean, pe_std=pe_std,
                ser_mean=ser_mean, ser_std=ser_std,
                alive=alive)


def build_one(pid: str, derived: Path, registered: Path) -> dict | None:
    t0_dir = derived / pid / "T0"
    if not t0_dir.exists():
        print(f"  {pid}: missing T0 derived/")
        return None

    t0_sv = np.load(t0_dir / "supervoxel_labels.npz")["labels"]
    t0_meta = json.loads((t0_dir / "meta.json").read_text())
    n_sv = int(t0_sv.max())
    if n_sv == 0:
        print(f"  {pid}: no supervoxels at T0")
        return None

    # T0 stats (from native data)
    t0_npz = np.load(t0_dir / "pe_ser.npz")
    spacing0 = t0_meta["voxel_spacing_mm"]
    origin0  = t0_meta["origin_mm"]
    stats_per_visit = {
        "T0": per_label_stats(
            t0_sv, t0_npz["pe"], t0_npz["ser"],
            spacing0, origin0, n_sv,
        )
    }

    # For each T_k follow-up: load the transported labels (in T_k native space)
    # and the T_k native PE/SER, then compute stats.
    for v in ("T1", "T2", "T3"):
        tk_dir = derived / pid / v
        trans_path = registered / pid / f"{v}_t0sv_in_{v}_space.npz"
        if not (tk_dir.exists() and trans_path.exists()):
            print(f"  {pid}: missing {v} or transport file")
            return None
        tk_meta = json.loads((tk_dir / "meta.json").read_text())
        tk_npz = np.load(tk_dir / "pe_ser.npz")
        trans = np.load(trans_path)
        tk_labels = trans["labels"]
        # Ensure label volume and PE volume share shape
        if tk_labels.shape != tk_npz["pe"].shape:
            print(f"  {pid}: shape mismatch at {v}: labels {tk_labels.shape}, pe {tk_npz['pe'].shape}")
            return None
        stats_per_visit[v] = per_label_stats(
            tk_labels, tk_npz["pe"], tk_npz["ser"],
            tk_meta["voxel_spacing_mm"], tk_meta["origin_mm"], n_sv,
        )

    # Stack into a single graph object.
    N = n_sv
    pos_list, x_list, t_list, node_id_list, alive_list = [], [], [], [], []
    visit_centroids = []   # per-visit tumor centroid (world mm) for diagnostics
    spacing_prod = float(np.prod(t0_meta["voxel_spacing_mm"]))  # mm^3 per voxel
    for v_idx, v in enumerate(VISITS):
        s = stats_per_visit[v]
        # Position fallback for dropped supervoxels: use T0 centroid so spatial
        # edges remain well-defined (alive flag will mask them out for losses).
        pos_v = s["centroids"].copy()
        dead = ~s["alive"]
        if dead.any():
            pos_v[dead] = stats_per_visit["T0"]["centroids"][dead]

        # ----------------------------------------------------------------
        # Subtract the per-visit tumor centroid (mean of alive nodes).
        # This removes patient repositioning (20–80 mm inter-visit DICOM
        # origin shifts) so the model only sees intra-tumour relative
        # displacements.  The centroid is stored in visit_centroids so we
        # can reconstruct world-mm positions for visualisation.
        # ----------------------------------------------------------------
        alive_mask = s["alive"]
        if alive_mask.sum() > 0:
            centroid_v = pos_v[alive_mask].mean(axis=0)
        else:
            centroid_v = pos_v.mean(axis=0)
        visit_centroids.append(centroid_v.tolist())
        pos_v = pos_v - centroid_v  # now centroid-relative

        feats = np.stack([
            np.log1p(s["voxel_count"].astype(np.float32)),
            (s["voxel_count"].astype(np.float32) * spacing_prod / 1000.0),
            s["pe_mean"], s["pe_std"],
            s["ser_mean"], s["ser_std"],
        ], axis=1)  # (N, 6)

        pos_list.append(pos_v)
        x_list.append(feats)
        t_list.append(np.full((N, 1), v_idx, dtype=np.float32))
        node_id_list.append(np.arange(N, dtype=np.int64))
        alive_list.append(s["alive"].astype(np.float32))

    pos = torch.from_numpy(np.concatenate(pos_list, axis=0)).float()
    x   = torch.from_numpy(np.concatenate(x_list,   axis=0)).float()
    t   = torch.from_numpy(np.concatenate(t_list,   axis=0)).float()
    nid = torch.from_numpy(np.concatenate(node_id_list, axis=0)).long()
    alive = torch.from_numpy(np.concatenate(alive_list, axis=0)).float()

    visit_offsets = torch.tensor(
        [0, N, 2 * N, 3 * N, 4 * N], dtype=torch.long
    )

    # Compute alive transitions for diagnostics
    alive_per_visit = alive.view(4, N)
    survival_rates = (alive_per_visit.mean(dim=1)).tolist()

    return dict(
        patient_id=pid,
        n_supervoxels=N,
        visit_ids=list(VISITS),
        x=x, pos=pos, t=t,
        node_id=nid,
        visit_offsets=visit_offsets,
        alive=alive,
        visit_centroids=visit_centroids,  # (4, 3) world-mm, for viz reconstruction
        feature_names=FEATURE_NAMES,
        survival_rate_per_visit=survival_rates,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient-list", type=Path,
                    default=None,
                    help="File with one patient ID per line. "
                         "Defaults to reports/prototype_patients.txt when --patient is not set.")
    ap.add_argument("--patient", type=str, default=None,
                    help="Single patient ID to process (alternative to --patient-list).")
    ap.add_argument("--derived", type=Path,
                    default=Path("datasets/ispy2/derived_v2"))
    ap.add_argument("--registered", type=Path,
                    default=Path("datasets/ispy2/registered"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("datasets/ispy2/graphs_consistent"))
    args = ap.parse_args()

    if args.patient:
        pids = [args.patient.strip()]
    else:
        list_path = args.patient_list or Path("reports/prototype_patients.txt")
        pids = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(pids)} consistent graphs -> {args.out_root}")

    n_ok, n_fail = 0, 0
    for i, pid in enumerate(pids):
        print(f"\n[{i+1}/{len(pids)}] {pid}")
        try:
            g = build_one(pid, args.derived, args.registered)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            n_fail += 1; continue
        if g is None:
            n_fail += 1; continue
        out_path = args.out_root / f"{pid}.pt"
        torch.save(g, out_path)
        sr = g["survival_rate_per_visit"]
        print(f"  N={g['n_supervoxels']}  survival(T0..T3): "
              f"{sr[0]:.2f} {sr[1]:.2f} {sr[2]:.2f} {sr[3]:.2f}  -> {out_path.name}")
        n_ok += 1

    print(f"\nBuilt {n_ok} graphs, {n_fail} failures")


if __name__ == "__main__":
    main()
