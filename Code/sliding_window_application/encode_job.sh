#!/bin/bash
#SBATCH -p ei-medium
#SBATCH -c 1 
#SBATCH -x t128n57
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -o slurm.encode.%N.%j.out # STDOUT
#SBATCH -e slurm.encode.%N.%j.err # STDOUT

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/encode_hpc.py --write_path ${1} --dna_seq till${2}${3} --encoded_seq encoded_till${2}${4}

# $1 - write path - raw/by100k_step25_chr12/
# $2 - a number of the chunk
# $3 -- suffix for this chr batch dna seq : 00k_by100k_dna_seq_25step.txt, or m_by1mil_dna_seq_1step.txt
# $4 -- suffix for encoded seq: 00k_by100k_s25, or m_by1mil_s1