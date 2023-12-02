#!/bin/bash -l
#SBATCH --job-name=examplejob   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=0-02:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000872  # Project for billing

module load LUMI/22.08
module load cotainr
cotainr build lumi_env_container_test.sif --base-image=docker://rocm/dev-ubuntu-22.04:5.3.2-complete --conda-env=lumi_env.yml
