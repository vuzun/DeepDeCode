#!/bin/bash
#SBATCH -p ei-medium
#SBATCH -c 1 
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -o slurm.subsetting.%N.%j.out # STDOUT
#SBATCH -e slurm.subsetting.%N.%j.err # STDERR

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/subsetting_dna_seq.py ${1} ${2} ${3} ${4}

# $1 DNA SEQ: raw/chr12_100by1_prepped.txt
# $2 CHUNK SIZE: 1000000
# $3 OUTPUTDIR (no / at end): data/best_folder
# $4 STEP SIZE: 1
