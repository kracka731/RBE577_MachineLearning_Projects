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

2. Replace 12.1 with your NVIDIA GPU version (if it works).
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install other packages 
```bash 
pip install robosuite
pip install robomimic
```

These packages can be seen at https://github.com/ARISE-Initiative/robosuite.git and https://github.com/ARISE-Initiative/robomimic.git 

## Verify Installation 
``` bash
python -c "import torch; print(torch.__version__)"
python -c "import robomimic; print('robomimic OK')"
python -c "import robosuite; print('robosuite OK')"
```