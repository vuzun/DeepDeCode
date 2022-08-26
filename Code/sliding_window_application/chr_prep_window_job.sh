#!/bin/bash
#SBATCH -p ei-medium
#SBATCH -c 1 
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -o slurm.chrprep.%N.%j.out # STDOUT
#SBATCH -e slurm.chrprep.%N.%j.err # STDERR

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/chr_prep_window.py ${1} ${2} ${3} ${4}

# $1 chr txt file, e.g. raw_data/txt/ch12.txt
# $2 chr chopped output (one big) file: chr12prepared.txt
# $3 subsequence length, how many nucleotides per line: 100
# $4 step length, NT difference between 2 adjacent lines: 25
