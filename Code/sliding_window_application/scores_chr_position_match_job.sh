#!/bin/bash
#SBATCH -p ei-medium # queue
#SBATCH -c 1 # number of cores
#SBATCH --mem 64G # memory pool for all cores

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/scores_chr_position_match.py $1 $2 $3 $4 $5 $6 $7 $8

# $1 chunk out of X that are parts of a chromosome split by some size (100k latest)
# $2 input file location
# $3 chunk size
# $4 step size
# $5 subseq length
# $6 chr number
# $7 optional folder name
# $8 optional annotation file (maybe shouldn't be optional)

# assumes chunk size, step and sequence length -- 100000 25 100
# also assumes prepared annotation for the chromosome!