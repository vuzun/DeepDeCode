#!/bin/bash
#SBATCH -p ei-medium # queue
#SBATCH -c 1 # number of cores
#SBATCH --mem=175GB
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

source package e919ef4d-be96-49b6-9bad-0a6b89ece048
srun pyTorch src/applying_deepdecode_by_chunks.py $1 $2 $3

# $1 - encodedseq file to score
# #files of interest are of the form 'till100k_byX00k_dna_seq_25step' with X from 1 to 23(or so)
# $2 - output location
# $3 - checkpoint file

#chr_encoded_seq="raw/by100k_step25_chr"+arg2+"/encoded_till"+arg1+"00k_by100k_s25"
#output_path="results/output_chr"+arg2+"_till"+arg1+"00k_by100k_25s.pt"
#checkpoint_path="raw/new_checkpoint_new_76_iter.pth"
