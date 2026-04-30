#!/bin/bash
#SBATCH --job-name=conda_test_job
#SBATCH --mem=20G
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=25             # Number of CPU cores per task
#SBATCH --gpus=1
#SBATCH --time=0:10:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=JKuehne@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3


# module load python
module load miniconda3/22.11.1/ygt4bhf

source $(conda info --base)/etc/jkuehne.d/conda.sh

conda config --append channels conda-forge
conda config --append channels pytorch
conda config --append channels nvidia

# conda init bash
# source /home/jkuehne/.bashrc

conda create -n robomimic_env python=3.8.20 --yes
conda activate robomimic_env

# conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install torch==2.0.0+cu121 torchvision==0.15.0+cu121 torchaudio==2.0.0 \
#   --index-url https://download.pytorch.org/whl/cu121
  
# # Setup Mujoco 2.1.0 binary (required by mujoco-py==2.1.2.14 in conda install file)
# mkdir -p ~/.mujuco
# wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
#   -O ~/.mujoco/mujoco210.tar.gz
# tar -xzf ~/.mujoco/mujoco210.tar.gz -C ~/.mujoco/
# rm ~/.mujoco/mujoco210.tar.gz

# Option A: Stay on torch 2.0.0 with CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 \
  --index-url https://download.pytorch.org/whl/cu118

# # Option B: Use torch 2.1.0 with CUDA 12.1 (recommended if cluster has CUDA 12)
# pip install torch==2.1.0+cu121 torchvision==0.15.0+cu121 torchaudio==2.1.0 \
#   --index-url https://download.pytorch.org/whl/cu121

mkdir -p ~/.mujoco
wget --server-response -q \
  https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -O ~/.mujoco/mujoco210.tar.gz \
  || { echo "wget failed — trying curl"; \
       curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
       -o ~/.mujoco/mujoco210.tar.gz; }

# Only extract if download succeeded
if [ -f ~/.mujoco/mujoco210.tar.gz ]; then
  tar -xzf ~/.mujoco/mujoco210.tar.gz -C ~/.mujoco/
  rm ~/.mujoco/mujoco210.tar.gz
else
  echo "ERROR: MuJoCo download failed — check cluster network access"
  exit 1
fi

# Set Mujoco hard-coded paths
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH   # for GPU rendering if available

# pip install -r requirements.txt

srun git clone git@github.com:kracka731/RBE577_MachineLearning_Projects.git

pip install -r req_freeze.txt

# conda install -n 


# conda config --append channels conda-forge
# conda create -n 'robomimic_env' --file conda-package-list.txt
# source /home/jkuehne/anaconda3/etc/profile.d/conda.sh
# conda init bash
# source /home/jkuehne/.bashrc
# conda activate robomimic_env

echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

# python hello_conda_script.py


