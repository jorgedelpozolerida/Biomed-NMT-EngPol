#!/bin/bash -l
#SBATCH --job-name=test   # Job name
#SBATCH --output=test.o%j # Name of stdout output file
#SBATCH --error=test.e%j  # Name of stderr error file
#SBATCH --partition=debug # Partition (queue) name
#SBATCH --nodes=1               # Total number of nodes
#SBATCH --ntasks=1
#SBATCH --time=0-02:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_46500087