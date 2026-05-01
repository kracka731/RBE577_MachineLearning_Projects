#!/bin/bash
#SBATCH --job-name=bc_test
#SBATCH --mem=30G
#SBATCH --output=bc_test_out.txt           # Standard output file
#SBATCH --error=bc_test_error.txt             # Standard error file
##SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=40             # Number of CPU cores per task
#SBATCH --gpus=2
#SBATCH --time=5:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=kracka@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load apptainer

apptainer run --userns ~/Project_4/Slurm/box.sif
ls -a
pip install robomimic 
pip install robosuite==1.4.1 --force-reinstall

python3 -c "import torch; print(torch.__version__)"
python3 -c "import robomimic; print(f'robomimic OK {robomimic.__version__}')"
python3 -c "import robosuite; print(f'robosuite OK {robosuite.__version__}')"

exit

apptainer exec --userns box.sif bash -c \ 'cp -r /usr/local/lib/python3.8/dist-packages/mujoco_py /tmp/mujoco_py_writable'
apptainer exec --userns --nv \
  --bind /tmp \
  --bind /tmp/mujoco_py_writable:/usr/local/lib/python3.8/dist-packages/mujoco_py \
  box.sif \
  python3 /root/workspace/Project_4/submodules/robomimic/examples/train_bc_rnn.py







