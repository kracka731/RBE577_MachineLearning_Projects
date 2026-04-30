#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --mem=4G
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
##SBATCH --partition=partition_name    # Partition or queue name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=0:02:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=kracka@wpi.edu    # Email address for notifications
#SBATCH -A rbe577 # for RBE577 P3
#SBATCH -p academic # for RBE577 P3
#SBATCH --mem=4G
module load python
python Slurm/hello_script.py
