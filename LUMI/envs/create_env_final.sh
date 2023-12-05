#!/bin/bash -l
#SBATCH --job-name=create_env   # Job name
#SBATCH --output=logs/create_env_output_%j # Name of stdout output file
#SBATCH --error=logs/create_env_error_%j  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --account=project_465000872  # Project for billing
#SBATCH --time=0-20:00:00       # Run time (d-hh:mm:ss)

module load LUMI/22.08
# module load LUMI/22.08 partition/G
module load cotainr
cotainr build final_container.sif  --base-image=docker://rocm/dev-ubuntu-22.04:5.3.2-complete --conda-env=final-env.yml