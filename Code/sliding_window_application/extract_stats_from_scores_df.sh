#!/bin/bash
#SBATCH -p ei-medium # queue
#SBATCH -c 1 # number of cores
#SBATCH --mem 64G # memory pool for all cores

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/extract_stats_from_scores_df.py "$@"

# $1 directory with scores df
# $2 (optional) threshold, default 0.8
# $3 (optional) output directory
