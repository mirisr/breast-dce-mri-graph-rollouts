# Environment Notes

The current local analysis environment used Python 3.13 with PyTorch 2.11 and
CPU execution for figure generation. GPU execution is only needed for full
training and large conditional MC runs.

For local analysis:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements-minimal.txt
```

For Cradle runs, use the Slurm wrappers under:

```text
experiments/stage1_forecaster/
experiments/consistent_rollout/
```

The retained MC runs were submitted to `gpul40q`.

