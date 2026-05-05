# Running on Cradle (UTRGV HPC)

This directory holds the setup and job-submission helpers for running the
3DGCNN project on the [Cradle HPC cluster](https://hpc.utrgv.edu).

**TL;DR commands:** see [`CRADLE_CHEATSHEET.md`](CRADLE_CHEATSHEET.md).

## Prerequisites

1. **Cradle account** — request via UTRGV IT's *Get Access!* portal.
2. **UTRGV VPN** — installed and active on your Mac whenever you SSH in.
3. **SSH key auth (recommended)** — skip daily password prompts:
   ```bash
   ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
   ssh-copy-id -i ~/.ssh/id_ed25519 <user>@login.cradle.utrgv.edu
   cat >> ~/.ssh/config <<'EOF'
   Host cradle
     HostName login.cradle.utrgv.edu
     User <user>
     IdentityFile ~/.ssh/id_ed25519
     ServerAliveInterval 60
   EOF
   chmod 600 ~/.ssh/config
   ```
   Then: `ssh cradle` works without a password.

## One-time setup

From the Cradle login node:

```bash
git clone https://github.com/mirisr/3DGCNN.git ~/3DGCNN
bash ~/3DGCNN/cradle/setup_cradle.sh
```

`setup_cradle.sh` is idempotent — you can re-run it to pick up missing
packages after a pull.

What it does:

- Installs Miniforge into `~/miniforge3` (no sudo, ~400 MB).
- Creates a conda env `3dgcnn` with Python 3.11.
- Installs **PyTorch 2.5.1 (cu121)** and **torch-geometric 2.6.1** plus the
  companion libraries `torch-scatter`, `torch-sparse`, `torch-cluster`.
- Installs the project's domain dependencies (numpy, pandas, scipy,
  scikit-learn, scikit-image, nibabel, pydicom, matplotlib, seaborn, jupyter,
  nbconvert, etc.).
- Submits a short `srun` job to `gpul40q` to verify that PyTorch can see an
  L40S GPU and run `CGConv` / `TransformerConv` on it.

Expected runtime: ~10–15 minutes, most of which is downloading wheels.

## Daily workflow

```bash
ssh cradle
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
# … edit scripts, submit jobs …
```

In a SLURM job script, you almost always want:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
module load cuda/12.3    # optional; PyTorch bundles its own cudart
```

## Cluster cheat sheet

GPU partitions available on Cradle (as of 2026-04):

| Partition  | GPU     | GPUs/node | Nodes | Walltime cap |
|------------|---------|-----------|-------|--------------|
| `gpul40q`  | L40S    | 8         | 2     | none         |
| `sxmq`     | SXM     | 8         | 4     | none         |
| `kimq`*    | SXM     | 8         | 2     | 24 h         |
| `gpua30q`  | A30     | 6         | 2     | 36 h         |
| `gpuq`     | mixed   | 8         | 8     | none         |
| `gpunvq`   | mixed   | 8         | 1     | none         |

We use **`gpul40q`** by default. Each L40S has **46 GB** usable VRAM.

Submit a job:

```bash
sbatch cradle/slurm/<jobfile>.sh
squeue -u $USER         # watch the queue
tail -f logs/*.<jobid>  # inspect outputs
scancel <jobid>         # cancel
```

## Pinned stack

| Component    | Version    |
|--------------|------------|
| OS           | Rocky Linux 8.6 |
| CUDA module  | 12.3       |
| Python       | 3.11       |
| PyTorch      | 2.5.1+cu121 |
| torchvision  | 0.20.1+cu121 |
| torch-geometric | 2.6.1    |
| torch-scatter/-sparse/-cluster | built against torch 2.5 + cu121 |

### Why these versions?

- **PyTorch 2.5 + CUDA 12.1 wheels** install cleanly, contain SASS for
  compute capabilities through `sm_90` plus forward-compatible PTX, so they
  run on the L40S (`sm_89`) after a one-time JIT compile on first kernel
  launch.
- **torch-geometric 2.6** has pre-built `torch-scatter`/`torch-sparse`/
  `torch-cluster` wheels at `data.pyg.org/whl/torch-2.5.0+cu121.html`,
  which avoids the painful source-build path.
- **Python 3.11** is inside PyTorch's and PyG's supported range for these
  versions, and is well ahead of the system Python 3.6 on the login node.

## Notes and gotchas

- **Login node has no GPU driver.** `torch.cuda.is_available()` is `False`
  on `login001` and will always be. Test GPU code via `srun` or in a SLURM
  batch job.
- **First kernel launch is slow** (a few seconds) because CUDA JIT-compiles
  PTX for `sm_89`. Subsequent launches in the same process are fast.
- **`/home` is shared NFS** (`riogrande:/home`) — OK for code, caches, and
  small datasets. For the I-SPY 2 DICOMs we will ask HPC for a project/
  scratch allocation rather than dumping tens of GB into our home quota.
- **Do not use the shared `/shared/pytorch-1.10.2/pytorch_env`.** It ships
  PyTorch 1.10 + CUDA 11.6 on Python 3.6, which is too old for our
  `torch-geometric` APIs and does not include kernels for the L40S.
