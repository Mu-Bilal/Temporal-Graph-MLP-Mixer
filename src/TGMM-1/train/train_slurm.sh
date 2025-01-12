#!/bin/bash

#SBATCH -J TGMM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=04:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/mnt/cephfs/store/gr-mc2473/lc865/workspace/GNN/slurm-logs/%x-%j.out

# Load environment (activate conda environment)
source /mnt/cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml-gnn

# Working directory
workdir="/home/lc865/workspace/DL-GNNs/Temporal-Graph-MLP-Mixer"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Run the Python script using the options
CMD="srun python src/TGMM-1/train/train.py"

echo "Executing command: $CMD"
eval $CMD