# Setup Instructions

## Check GPU version
```bash 
nvidia-smi
``` 

## Set up Environment 
1. Create Env
```bash 
conda create -n robomimic_env python=3.8 
conda activate robomimic_env
```

2. Replace 12.1 with your NVIDIA GPU version or simply use 12.1.
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Use Git Submodules 
```bash 
git submodule init
git submodule update --recursive
```

In the robosuite directory, run: 
```bash 
pip install -e .
pip install -r submodules/robosuite/requirements-extra.txt 
```

In the robomimic directory, run: 
```bash 
pip install -e .
```

4. Install required packages 
```bash
pip install robosuite
pip install mujoco-py
pip install "Cython<3"
sudo apt install patchelf
```

## Verify Installation 
``` bash
python -c "import torch; print(torch.__version__)"
python -c "import robomimic; print('robomimic OK')"
python -c "import robosuite; print('robosuite OK')"
```

You can also run the following test file. It does not need to finish training for you to know its working.
```bash 
python Project_4/submodules/robomimic/examples/train_bc_rnn.py 
```