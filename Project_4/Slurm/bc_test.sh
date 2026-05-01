#!/bin/bash
#SBATCH --job-name=bc_training
#SBATCH --mem=40G
#SBATCH --output=bc_test_out.txt           # Standard output file
#SBATCH --error=bc_test_error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=40             # Number of CPU cores per task
#SBATCH --gpus=2
#SBATCH --time=0-16:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=kracka@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load apptainer

# apptainer run --userns ~/Project_4/Slurm/box.sif # use this when interactive
# apptainer exec --userns ~/Project_4/Slurm/box.sif bash

ls -a
# apptainer exec --userns ~/Project_4/Slurm/box.sif bash -c \
# apptainer exec --userns ~/Project_4/Slurm/box.sif bash

apptainer exec --userns ~/Project_4/Slurm/box.sif pip install robomimic
apptainer exec --userns ~/Project_4/Slurm/box.sif pip install robosuite==1.4.1 --force-reinstall
apptainer exec --userns ~/Project_4/Slurm/box.sif \
    python3 -c "import torch; print(torch.__version__)"
apptainer exec --userns ~/Project_4/Slurm/box.sif \
    python3 -c "import robomimic; print(f'robomimic OK {robomimic.__version__}')"

apptainer exec --userns ~/Project_4/Slurm/box.sif \
    python3 -c "import robosuite; print(f'robosuite OK {robosuite.__version__}')"


apptainer exec --userns ~/Project_4/Slurm/box.sif bash -c \ 'cp -r /usr/local/lib/python3.8/dist-packages/mujoco_py /tmp/mujoco_py_writable'
apptainer exec --userns --nv \
  --bind /tmp \
  --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
  ~/Project_4/Slurm/box.sif \
  bash -c 'yes | python3 /root/workspace/Project_4/submodules/robomimic/examples/train_bc_rnn.py'







