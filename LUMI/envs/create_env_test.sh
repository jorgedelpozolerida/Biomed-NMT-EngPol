#!/bin/bash -l
#SBATCH --job-name=examplejob   # Job name
#SBATCH --output=logs/examplejob_output_%j # Name of stdout output file
#SBATCH --error=logs/examplejob_error_%j  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --account=project_465000872  # Project for billing
#SBATCH --time=0-10:00:00       # Run time (d-hh:mm:ss)

module load LUMI/22.08
# module load LUMI/22.08 partition/G
module load cotainr
cotainr build test_container_v3.sif  --base-image=docker://rocm/dev-ubuntu-22.04:5.3.2-complete --conda-env=test-env.yml