#! /usr/bin/env python3

import subprocess
from subprocess import CalledProcessError

"""
Master script to run a full WandB sweep. Simply call this using Python (no sbatch/srun needed).

Workflow:
1. Creates WandB sweep with parameters specified in `config_path` and extracts sweep path (entity/project/sweep-id).
2. Schedules `job_count` jobs to run the sweep agent in parallel.

Scripts/files that need to be configured:
- `slurm_sweep_agent.sh`: Script to run a single sweep agent via slurm. Invoked via `sbatch` in this script (`JOB_COUNT` times).
- `src/sweeps/train_wrapper.py`: Script to run a single training job. Called by the WandB agent and copies the hyperparameters for that particular run via `wandb.config`, which is made available when a WandB agent calls the script.
- `src/model/train.py`: Training script that is called by `train_wrapper.py`. This contains the main training loop.

- `src/sweeps/sweep_cfg.yaml`: Config file containing sweep parameters, schedule (e.g. grid search) and reference to `train_wrapper.py` script.
- `src/model/config.yaml`: Default config file with non-sweep parameters.

"""

CONFIG_PATH = '/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/sweep/sweep_config.yaml'
PROJECT_NAME = 'GMM-1'
N_RUNS = 1
N_ARRAY_RUNS = 4  # Preferred to multiple runs.
RUNS_PER_AGENT = 20


# Execting wandb in CLI instead of using python API as the latter somehow does not close the localhast connection resulting in an error when starting the agent
print(f'Creating sweep from config: {CONFIG_PATH}')
try:
    result = subprocess.run(f'wandb sweep --project {PROJECT_NAME} {CONFIG_PATH}', shell=True, capture_output=True, text=True, check=True)
except CalledProcessError as e:
    print(f'Error creating sweep. Is the config file path correct? Original error message: \n {e.stderr}')
    raise

sweep_path = result.stderr.split("wandb: Run sweep agent with: wandb agent ")[1].strip()  # NB: For some reason the output is in stderr
assert len(sweep_path) > 0

print(result.stderr)
print(f'Extracted sweep path: {sweep_path}')

command = f'cd /home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer/src/TGMM-1/sweep && sbatch --array=0-{N_ARRAY_RUNS-1} slurm_sweep_agent.sh {sweep_path} {RUNS_PER_AGENT}'

try:
    for i in range(N_RUNS):
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f'Scheduled job {i}: {result.stdout}')
except CalledProcessError as e:
    print(f'Error scheduling jobs. Original error message: \n {e.stderr}')
    raise
