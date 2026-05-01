#!/bin/bash
#SBATCH -A rbe577 ## for RBE577 P4
#SBATCH -p academic ## for RBE577 P4
#SBATCH --job-name "container_setup"
#SBATCH --mem=50G
#SBATCH --output=container_setup_out.txt           # Standard output file
#SBATCH --error=container_setup_error.txt             # Standard error file
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task 25             # Number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --time 0-01:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=kracka@wpi.edu 

module load apptainer 

apptainer build box.sif Slurm/apptainer_def.def
