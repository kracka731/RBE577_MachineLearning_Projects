#!/bin/bash
#SBATCH --job-name=config_gen
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

cd /home/jkuehne/Project_4

module load apptainer


apptainer exec --userns Slurm/box.sif python3 << 'EOF'
import robomimic
from robomimic.config import config_factory
import json

# Create BC config for your installed version
config = config_factory(algo_name='bc')

# Basic training settings
config.train.data = None  # Will be set via command line
config.train.output_dir = 'bc_trained_models'
config.train.num_epochs = 2000
config.train.batch_size = 100
config.train.seq_length = 1

# Experiment settings
config.experiment.name = 'BC_Cloning_Experiment'
config.experiment.rollout.enabled = True
config.experiment.rollout.n = 50
config.experiment.rollout.horizon = 400
config.experiment.rollout.rate = 50
config.experiment.rollout.terminate_on_success = True
config.experiment.save.enabled = True
config.experiment.save.every_n_epochs = 50
config.experiment.save.on_best_rollout_success_rate = True
config.experiment.render_video = True

# Algorithm settings
config.algo.optim_params.policy.learning_rate.initial = 0.0001
config.algo.actor_layer_dims = [1024, 1024]

# Observation settings
config.observation.modalities.obs.low_dim = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']

# Save to file
with open('training_configs/bc_clean.json', 'w') as f:
    json.dump(config, f, indent=4)
    
print("✓ Generated bc_clean.json compatible with your robomimic version")
print(f"  Robomimic version: {robomimic.__version__}")
EOF