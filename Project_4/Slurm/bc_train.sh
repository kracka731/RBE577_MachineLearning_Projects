#!/bin/bash
#SBATCH --job-name=bc_training
#SBATCH --mem=40G
#SBATCH --output=bc_train_out.txt           # Standard output file
#SBATCH --error=bc_train_error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=40             # Number of CPU cores per task
#SBATCH --gpus=2
#SBATCH --time=0-16:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=jkuehne@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load apptainer

PROJECT_DIR=${SLURM_SUBMIT_DIR}

echo "Working from: ${PROJECT_DIR}"

# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif pip install robomimic
# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif pip install robosuite==1.4.1 --force-reinstall
# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import torch; print(torch.__version__)"
# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import robomimic; print(f'robomimic OK {robomimic.__version__}')"

# apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif \
#     python3 -c "import robosuite; print(f'robosuite OK {robosuite.__version__}')"


apptainer exec --userns ${PROJECT_DIR}/Slurm/box.sif bash -c \ 'cp -r /usr/local/lib/python3.8/dist-packages/mujoco_py /tmp/mujoco_py_writable'

apptainer exec --userns \
  --bind ${PROJECT_DIR}:/work \
  ${PROJECT_DIR}/Slurm/box.sif \
  bash -c "cp /work/demonstrations/merged_converted.hdf5 /root/workspace/Project_4/demonstrations/ && \
           cp /work/training_configs/bc_clean.json /root/workspace/Project_4/training_configs/"



apptainer exec --userns --nv \
  --bind /tmp \
  --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
  --bind ${PROJECT_DIR}:/work \
  --bind ${PROJECT_DIR}/bc_trained_models:/root/workspace/Project_4/bc_trained_models \
  ${PROJECT_DIR}/Slurm/box.sif \
  bash -c "yes | python3 -m /root/workspace/Project_4/submodules/robomimic/robomimic/scripts/train.py \
  --config /root/workspace/training_configs/bc_clean.json \
  --dataset /root/workspace/demonstrations/merged_converted.hdf5 \
  --name 'BC_Cloning_Experiment'"

# Copy results back to your directory
apptainer exec --userns \
  --bind ${PROJECT_DIR}:/work \
  ${PROJECT_DIR}/Slurm/box.sif \
  bash -c "cp -r /root/workspace/Project_4/bc_trained_models/* /work/bc_trained_models/"

echo "Training complete! Results in bc_trained_models/"

