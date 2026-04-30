#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=00:00:10
#SBATCH --mem=4G
module load python
python slurm_test.sh
