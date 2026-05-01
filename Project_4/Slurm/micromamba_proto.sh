#!/bin/bash
#SBATCH --job-name=run_micromamba_test_proto
#SBATCH --mem=30G
#SBATCH --output=proto_mamba_output.txt           # Standard output file
#SBATCH --error=proto_mamba_error.txt             # Standard error file
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

# module load miniconda3/22.11.1/ygt4bhf

# setup micromamba
export MAMBA_ROOT_PREFIX=~/.micromamba   # stores envs in your home dir
eval "$(~/.local/bin/micromamba shell hook -s bash)"

module load libx11/1.7.0/hlcc3e6
# Expose X11 headers for mujoco-py compilation
export CPATH=$LIBX11_ROOT/include:$CPATH
export MUJOCO_GL=egl

module load glew/2.2.0/azi6l2x
# export CPATH=$OpenGL_ROOT/include:$CPATH


# conda clean --all -y
# pip cache purge
# conda deactivate
# conda env remove -n robomimic_env --yes
# conda config --set solver classic

# Remove any lingering environment

micromamba env remove -n robomimic_env --yes 2>/dev/null || true

# Also remove the directory manually in case micromamba left a partial folder
# (this was causing "Non-conda folder exists at prefix" in your first error)
rm -rf ~/.micromamba/envs/robomimic_env

# Reset channels to a known good state
# conda config --remove-key channels 2>/dev/null || true
conda config --add channels defaults
conda config --append channels conda-forge
conda config --append channels pytorch
conda config --append channels nvidia

# micromamba config --remove-key channels 2>/dev/null || true
# micromamba config --add channels defaults
# micromamba config append channels conda-forge
# micromamba config append channels pytorch
# micromamba config append channels nvidia

# ── Create and activate env ──────────────────────────────────

echo "=== Creating environment ==="
micromamba create -n robomimic_env python=3.8.20 --yes \
    -c conda-forge -c defaults -c pytorch -c nvidia
CREATE_EXIT=$?
echo "=== micromamba create exited with: $CREATE_EXIT ==="

# List what's in the env dir so we can see what was actually created
echo "=== Contents of env prefix ==="
ls -la ~/.micromamba/envs/robomimic_env/bin/ 2>/dev/null || echo "DIRECTORY DOES NOT EXIST"

if [ $CREATE_EXIT -ne 0 ]; then
    echo "FATAL: micromamba create failed — aborting"
    exit 1
fi

echo "activating environment"
micromamba activate robomimic_env

ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/robomimic_env"
PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"

$PIP install pyopengl


# echo "=== Checking python binary ==="
# ls -la $PY || echo "python3.8 not found at $PY"
# ls -la $ENV_PREFIX/bin/python* || echo "no python binaries found"

# ── Hard stop if wrong Python ─────────────────────────────────
PY_VER=$($PY --version 2>&1)
echo "Python: $PY_VER"
if [[ "$PY_VER" != *"3.8"* ]]; then
    echo "FATAL: Wrong Python $PY_VER — aborting"
    exit 1
fi
echo "Using: $($PY -c 'import sys; print(sys.executable)')"

# source ~/.conda/envs/robomimic_env/lib/python3.8/venv/scripts/common/activate
# source ~/.conda/envs/robomimic_env/bin/activate
# Find the correct conda.sh path
# MICROCAMBA_BASE=$(microcamba info --base)
# source $MICROCAMBA_BASE/etc/profile.d/conda.sh



# python --version

micromamba install -c conda-forge \
    brotli certifi charset-normalizer filelock idna \
    jinja2 markupsafe mpmath networkx pillow pysocks \
    requests sympy urllib3 gmpy2 libstdcxx-ng xorg-libx11 \
    mesa mesa-libgl-devel-cos7-x86_64 mesa-libglu-devel-cos7-x86_64 glew freegluts \
    conda-forge xorg-xproto xorg-libx11 glew mkl mkl-include mkl-service

# torch 2.0.0 with CUDA 11.8
$PIP install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0 \
  --index-url https://download.pytorch.org/whl/cu118
echo "torch installed"
# $PY --version
ls $MAMBA_ROOT_PREFIX/envs/robomimic_env/include/X11/X.h || exit 1

# micromamba install -c conda-forge xorg-libx11 xorg-libx11-devel -y
# micromamba install -n robomimic_env -c conda-forge libx11-devel -y
# OpenGL Extension Wrangler (needed by mujoco-py EGL backend)
rm -rf ~/.mujoco_py

# ls $MAMBA_ROOT_PREFIX/envs/robomimic_env/include/X11/X.h

# # Set Mujoco hard-coded paths
# export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
# export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH   # for GPU rendering if available
$PIP install "cython<3" "cython==0.29.37"
$PIP install mujoco-py==2.1.2.14 --no-cache-dir
mkdir -p ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
     -o ~/.mujoco/mujoco210.tar.gz
tar -xzf ~/.mujoco/mujoco210.tar.gz -C ~/.mujoco/
rm ~/.mujoco/mujoco210.tar.gz
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
echo "installed cython and mujoco-py"
$PY --version
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

source ~/.bashrc

# ls $MAMBA_ROOT_PREFIX/envs/robomimic_env/include/X11/Xlib.h

echo "commited the large install"
$PY --version


echo "installed mkl"
# $PY --version
    
source ~/.bashrc


$PIP install -r ~/scratch/Project_4/Slurm/pip_reqs_froze.txt
echo "installed pip freeze compilation"


# $PIP install -e ~/scratch/Project_4/submodules/robosuite
# echo "installed robosuite"
# $PY --version

# $PIP install -r ~/scratch/Project_4/submodules/robosuite/requirements-extra.txt 
# echo "and extra"
# # $PY --version

# $PIP install -e ~/scratch/Project_4/submodules/robomimic
# echo "installed robomimic"
# # $PY --version

$PIP install "pillow==9.5.0" --force-reinstall

# ── Install robosuite pinned to a version compatible with mujoco 2.x ──
# The current robosuite main requires mujoco>=3.3.0 which has no py3.8 wheel
# Pin to the last version that worked with mujoco-py / mujoco 2.x
$PIP install "robosuite==1.4.1" --no-deps
# Then install its deps manually (without the mujoco>=3.3.0 constraint)
$PIP install \
    "numpy==1.24.3" \
    "scipy" \
    "numba" \
    "matplotlib" \
    "h5py" \
    "imageio" \
    "imageio-ffmpeg" \
    "PyYAML"
echo "installed robosuite 1.4.1"

# ── Install robomimic ─────────────────────────────────────────
$PIP install -e ~/scratch/Project_4/submodules/robomimic --no-deps
$PIP install \
    "h5py>=3.1.0" \
    "numpy==1.24.3" \
    "tensorboard>=2.3.0" \
    "tensorboardX>=2.1"
echo "installed robomimic"

$PIP install "numpy==1.24.3" --force-reinstall

# pip install robosuite
# pip install mujoco-py
# pip install "Cython<3"
yes | $PY ~/scratch/Project_4/submodules/robosuite/robosuite/scripts/setup_macros.py                      

# python --version
# ── Hard stop if wrong Python ─────────────────────────────────
PY_VER=$($PY --version 2>&1)
echo "$PY: $PY_VER"
if [[ "$PY_VER" != *"3.8"* ]]; then
    echo "FATAL: Wrong Python $PY_VER — aborting"
    exit 1
fi
echo "Using: $(which $PY) at final portion"
$PY -c "import torch; print(torch.__version__)"
$PY -c "import robomimic; print('robomimic OK')"
$PY -c "import robosuite; print('robosuite OK')"


yes | $PY ~/scratch/Project_4/submodules/robomimic/examples/train_bc_rnn.py


