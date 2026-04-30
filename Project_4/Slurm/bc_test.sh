#!/bin/bash
#SBATCH --job-name=bc_test
#SBATCH --mem=30G
#SBATCH --output=bc_test.txt           # Standard output file
#SBATCH --error=bc_test_error.txt             # Standard error file
##SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=25             # Number of CPU cores per task
#SBATCH --gpus=1
#SBATCH --time=1:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=kracka@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load python=

# # python --version
# # ── Hard stop if wrong Python ─────────────────────────────────
# PY_VER=$($PY --version 2>&1)
# echo "$PY: $PY_VER"
# if [[ "$PY_VER" != *"3.8"* ]]; then
#     echo "FATAL: Wrong Python $PY_VER — aborting"
#     exit 1
# fi
# echo "Using: $(which $PY) at final portion"
# $PY -c "import torch; print(torch.__version__)"
# $PY -c "import robomimic; print('robomimic OK')"
# $PY -c "import robosuite; print('robosuite OK')"


python /submodules/robomimic/examples/train_bc_rnn.py





