import os
import pandas as pd
import sys

# making shorter lines of SUB_LENGTH with step of STEP_LENGTH

#SUB_LENGTH=100
#STEP_LENGTH=25
#chr_path="C:/Users/uzunv/repos/sampling_chromosome/raw_data/txt/chr12.txt"
#write_path="C:/Users/uzunv/repos/sampling_chromosome/raw_data/chr12_seq_25step.txt"

def split_into_subsequences(chr_path, write_path, SUB_LENGTH, STEP_LENGTH):
    file_object=open(chr_path, "r")
    chrm_seq=file_object.read() # X
    chrm_seq=chrm_seq.replace(" ","") # X-2

    dnaseq_file=open(write_path,"w")

    # less than 20
    for i in range(0,len(chrm_seq)-SUB_LENGTH, STEP_LENGTH):
        segment=chrm_seq[i:i+SUB_LENGTH]
        _=dnaseq_file.write("%s\n" % segment)
        if i%100000==0:
            print(i)

    dnaseq_file.close()

if __name__=="__main__":
    chr_file=sys.argv[1]
    write_path=sys.argv[2]
    sub_l=int(sys.argv[3])
    step_l=int(sys.argv[4])
    
    split_into_subsequences(chr_file, write_path, sub_l, step_l)