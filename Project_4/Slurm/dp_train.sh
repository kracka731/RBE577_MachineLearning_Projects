#!/bin/bash
#SBATCH --job-name=dp_training
#SBATCH --mem=40G
#SBATCH --output=Slurm/dp_train_out.txt           # Standard output file
#SBATCH --error=Slurm/dp_train_error.txt             # Standard error file
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

PROJECT_DIR=${SLURM_SUBMIT_DIR}
echo "Working from: ${PROJECT_DIR}"

# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif pip install robomimic
apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif pip install robosuite==1.5.1 --force-reinstall # may not be required. 
# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import torch; print(torch.__version__)"
# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import robomimic; print(f'robomimic OK {robomimic.__version__}')"

# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import robosuite; print(f'robosuite OK {robosuite.__version__}')"


# Clean any leftover from a previous run
rm -rf ${PROJECT_DIR}/mujoco_py_writable

# Copy mujoco_py into the project space
apptainer exec --userns \
  --bind ${PROJECT_DIR}:/work \
  ${PROJECT_DIR}/Slurm/box.sif \
  bash -c "cp -r /usr/local/lib/python3.8/dist-packages/mujoco_py /work/mujoco_py_writable"

# Train the model inside the box, using config and demonstration data from outside the box
echo "Starting training..."
apptainer exec --userns --nv \
  --bind ${PROJECT_DIR}:/work \
  --bind ${PROJECT_DIR}/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
  --env PYTHONPATH=/root/workspace/Project_4/submodules/robomimic:$PYTHONPATH \
  ${PROJECT_DIR}/Slurm/box.sif \
  bash -c "yes | python3 /root/workspace/Project_4/submodules/robomimic/robomimic/scripts/train.py \
  --config /work/training_configs/diffusion_policy.json \
  --dataset /work/demonstrations/merged_converted.hdf5 \
  --name 'DP_Cloning_Experiment'"



# apptainer exec --userns --nv \
#   --bind /tmp \
#   --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
#   --bind ${PROJECT_DIR}:/work \
#   ${PROJECT_DIR}/Slurm/box.sif \
#   bash -c "yes | python3 -m robomimic.scripts.train \
#   --config '/work/training_configs/diffusion_policy.json' \
#   --dataset '/work/demonstrations/merged_converted.hdf5' \
#   --name 'DP_Experiment'"

# apptainer exec --userns --nv \
#   --bind /tmp \
#   --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
#   --env PYTHONPATH=/root/workspace/Project_4/submodules/robomimic:$PYTHONPATH \
#   Slurm/box.sif \
#   bash -c 'yes | python3 /root/workspace/Project_4/submodules/robomimic/robomimic/scripts/train.py \
#     --config "/root/workspace/Project_4/training_configs/diffusion_policy.json" \
#     --dataset "/root/workspace/Project_4/demonstrations/merged_converted.hdf5" \
#     --name "DP_Experiment"'

# apptainer exec --userns --nv \
#   --bind /tmp \
#   --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
#   --bind ${PROJECT_DIR}:/work \
#   --env PYTHONPATH=/root/workspace/Project_4/submodules/robomimic:$PYTHONPATH \
#   Slurm/box.sif \
#   bash -c 'yes | python3 /root/workspace/Project_4/submodules/robomimic/robomimic/scripts/train.py \
#     --config "/root/workspace/Project_4/training_configs/diffusion_policy.json" \
#     --dataset "/root/workspace/Project_4/demonstrations/merged_converted.hdf5" \
#     --name "DP_Experiment"'
