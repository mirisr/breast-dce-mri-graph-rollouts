# Cradle quick reference (3DGCNN)

UTRGV [Cradle HPC](https://hpc.utrgv.edu). This condenses `cradle/README.md`, `setup_cradle.sh`, and how we submit jobs from `experiments/**/*.sbatch`.

---

## Before you SSH

| Step | Command / note |
|------|----------------|
| VPN | UTRGV VPN **on** (required from off-campus). |
| SSH | `ssh cradle` if `~/.ssh/config` has the `Host cradle` block from the README. |
| Repo path on cluster | We assume **`~/3DGCNN`** (matches sbatch scripts that `cd "$HOME/3DGCNN"`). |

---

## One-time setup (login node)

```bash
git clone https://github.com/mirisr/3DGCNN.git ~/3DGCNN
cd ~/3DGCNN
bash cradle/setup_cradle.sh
```

**What it installs:** Miniforge → `~/miniforge3`, conda env **`3dgcnn`** (Python 3.11), PyTorch **2.5.1+cu121**, torch-geometric **2.6.1** + scatter/sparse/cluster wheels, project pip deps, then an **`srun`** GPU smoke test on **`gpul40q`**.

**Override env vars (optional):**

| Variable | Default | Meaning |
|----------|---------|---------|
| `INSTALL_DIR` | `$HOME/miniforge3` | Conda root |
| `ENV_NAME` | `3dgcnn` | Conda env name |
| `GPU_PARTITION` | `gpul40q` | Partition for the smoke-test `srun` |

Re-run `setup_cradle.sh` anytime; it skips finished steps.

---

## Every new shell (login or job)

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
cd ~/3DGCNN
```

**Inside SLURM scripts** we usually mirror `experiments/**/*.sbatch`:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate 3dgcnn
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/3dgcnn/lib:${LD_LIBRARY_PATH:-}"
PY="$HOME/miniforge3/envs/3dgcnn/bin/python"
# optional:
# module load cuda/12.3
```

---

## GPU partitions (defaults)

| Partition | GPU | Notes |
|-----------|-----|--------|
| **`gpul40q`** | L40S (8/node, 46 GB usable) | **Default** for our sbatch files. |
| `sxmq` | SXM | |
| `kimq` | SXM | 24 h wall cap |
| `gpua30q` | A30 | 36 h wall cap |
| `gpuq` / `gpunvq` | mixed | |

---

## Submit jobs (from `~/3DGCNN`)

Our batch scripts live under **`experiments/`**, not under `cradle/slurm/` (the README example path is generic—you can add wrappers there later).

```bash
cd ~/3DGCNN
mkdir -p experiments/stage1_forecaster/logs   # if the sbatch references it

# Single job
sbatch experiments/stage1_forecaster/run_consistent_forecaster_5fold.sbatch

# Array job (uses #SBATCH --array=… inside the file)
sbatch experiments/stage1_forecaster/run_consistent_forecaster_5fold.sbatch

# Chain after another job finishes
sbatch --dependency=afterok:<JOBID> experiments/stage1_forecaster/run_consistent_forecaster_5fold.sbatch
```

**Watch / control**

```bash
squeue -u $USER
sacct -u $USER --format=JobID,JobName,Partition,State,Elapsed,MaxRSS
scancel <jobid>                    # whole job
scancel <array_job_id>_<task_id>    # one array task
tail -f experiments/stage1_forecaster/logs/cg5fold-<JOBID>_<TASK>.out
```

Log paths are **per script** (`#SBATCH --output=...`); grep the `.sbatch` file you use.

---

## Interactive GPU (debug)

Login node has **no** GPU driver—always test on a compute node:

```bash
srun -p gpul40q --gres=gpu:1 -t 00:30:00 --pty bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgcnn
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Pinned stack (matches `setup_cradle.sh`)

| Piece | Version |
|-------|---------|
| Python | 3.11 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 **+cu121** wheels |
| torch-geometric | 2.6.1 |
| CUDA module (optional) | `module load cuda/12.3` |

---

## Gotchas (from our README + scripts)

1. **`torch.cuda.is_available()` is False on the login node** — expected.
2. **First CUDA kernel on L40S can JIT a few seconds** (PTX → `sm_89`); then fast.
3. **`/home` is NFS** (`riogrande:/home`) — fine for code and modest data; huge DICOM mirrors belong on project/scratch if allocated.
4. **Do not use** the cluster shared `/shared/pytorch-1.10.2/pytorch_env` — too old for this repo.
5. **Job `cd`** — many sbatch files assume **`cd "$HOME/3DGCNN"`**; keep the clone there or edit the scripts.

---

## Example: where logs go (consistent forecaster 5-fold)

From `run_consistent_forecaster_5fold.sbatch`:

- stdout: `experiments/stage1_forecaster/logs/cg5fold-%A_%a.out`
- stderr: `experiments/stage1_forecaster/logs/cg5fold-%A_%a.err`

(`%A` = array job id, `%a` = task id.)

---

## Full narrative + SSH key setup

See **`cradle/README.md`**.
