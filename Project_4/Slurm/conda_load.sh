#!/bin/bash
#SBATCH --job-name=run_test_learning
#SBATCH --mem=30G
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=25             # Number of CPU cores per task
#SBATCH --gpus=1
#SBATCH --time=1:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=JKuehne@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3

module load miniconda3/22.11.1/ygt4bhf

# setup micromamba
xport MAMBA_ROOT_PREFIX=~/.micromamba   # stores envs in your home dir
eval "$(~/.local/bin/micromamba shell hook -s bash)"


module load libx11/1.7.0/hlcc3e6
# Expose X11 headers for mujoco-py compilation
export CPATH=$LIBX11_ROOT/include:$CPATH
export MUJOCO_GL=egl

conda clean --all -y
pip cache purge
conda deactivate
conda env remove -n robomimic_env --yes

conda config --set solver classic

# conda create -n robomimic_env python=3.8.20 --yes --no-default-packages
# ── Create and activate env ──────────────────────────────────
micromamba env remove -n robomimic_env --yes 2>/dev/null || true
micromamba create -n robomimic_env python=3.8.20 --yes \
    -c conda-forge -c defaults

micromamba activate robomimic_env

# ── Verify Python ────────────────────────────────────────────
PY_VER=$(python --version 2>&1)
echo "Python: $PY_VER"
if [[ "$PY_VER" != *"3.8"* ]]; then
    echo "FATAL: Wrong Python — aborting"
    exit 1
fi

# source $(conda info --base)/etc/jkuehne.d/conda.sh

# conda init bash
# source /home/jkuehne/.bashrc 
# conda init bash

# Reset channels to a known good state
conda config --remove-key channels 2>/dev/null || true
conda config --add channels defaults
conda config --append channels conda-forge
conda config --append channels pytorch
conda config --append channels nvidia





# conda create -n robomimic_env python=3.8.20 --yes
# conda activate robomimic_env
# source activate robomimic_env

# source ~/.conda/envs/robomimic_env/lib/python3.8/venv/scripts/common/activate
# source ~/.conda/envs/robomimic_env/bin/activate
# Find the correct conda.sh path
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# To speed up solving speed, ran in the base environment
# conda install -n base -c conda-forge mamba --yes
# conda install -n base conda-libmamba-solver
# conda config --set solver libmamba

# Now activate works
conda activate robomimic_env


python --version


# torch 2.0.0 with CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 \
  --index-url https://download.pytorch.org/whl/cu118


# # Setup Mujoco 2.1.0 binary (required by mujoco-py==2.1.2.14 in conda install file)
# mkdir -p ~/.mujoco
# wget --server-response -q \
#   https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
#   -O ~/.mujoco/mujoco210.tar.gz \
#   || { echo "wget failed — trying curl"; \
#        curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
#        -o ~/.mujoco/mujoco210.tar.gz; }

# # Only extract if download succeeded
# # if [ -f ~/.mujoco/mujoco210.tar.gz ]; then
# tar -xzf ~/.mujoco/mujoco210.tar.gz -C ~/.mujoco/
# rm ~/.mujoco/mujoco210.tar.gz
# else
#   echo "ERROR: MuJoCo download failed — check cluster network access"
#   exit 1
# fi

# # Set Mujoco hard-coded paths
# export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
# export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH   # for GPU rendering if available
pip install "cython<3" "cython==0.29.37"
pip install mujoco-py==2.1.2.14 --no-cache-dir

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


# New mujoco bindings (easier, no compilation)
# pip install mujoco==3.3.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

source ~/.bashrc


conda install -c conda-forge \
    brotli certifi charset-normalizer filelock idna \
    jinja2 markupsafe mpmath networkx pillow pysocks \
    requests sympy urllib3 gmpy2

# conda install -y -c defaults mkl-fft mkl-random mkl-service
conda install -y --strict-channel-priority -c defaults \
    mkl-fft=1.3.6 mkl-random=1.2.4 mkl-service=2.4.0
    
source ~/.bashrc

# srun git clone git@github.com:kracka731/RBE577_MachineLearning_Projects.git
# srun git clone https://github.com/kracka731/RBE577_MachineLearning_Projects.git 


# Using on Project_4 directory: pip list --format=freeze | sed 's/ @ file:\/\/.*//' \ | grep -v "^-e " \ > pip_reqs_froze.txt

# pip install xlib
# pip install --force-reinstall -v "numpy==1.24.3"
# pip install python-xlib
# srun git submodule init
# srun git submodule update --recursive




# yes | python xlib_setup.py install
cd ~/scratch/Project_4
git submodule init
git submodule update --recursive
# Per Project_4 Readme
cd submodules/robosuite
git tag | grep -i "1.5.1"          # find the exact tag name first
git checkout v1.5.1                 # or whatever the tag shows
cd ~/scratch/Project_4

pip install -e ~/scratch/Project_4/submodules/robosuite
pip install -r ~/scratch/Project_4/submodules/robosuite/requirements-extra.txt 
pip install -e ~/scratch/Project_4/submodules/robomimic

pip install -r pip_reqs_froze.txt

pip install "numpy==1.24.3" --force-reinstall

# pip install robosuite
# pip install mujoco-py
# pip install "Cython<3"
yes | python ~/scratch/Project_4/submodules/robosuite/robosuite/scripts/setup_macros.py

# echo "Python: $(which python)"
# echo "Env: $CONDA_DEFAULT_ENV"

# module load libx11/1.8.12/wtcqjwl
# pip install mink==0.0.5                           


python -c "import torch; print(torch.__version__)"
python -c "import robomimic; print('robomimic OK')"
python -c "import robosuite; print('robosuite OK')"

yes | python ~/scratch/Project_4/submodules/robomimic/examples/train_bc_rnn.py

# srun y

# python hello_conda_script.py


