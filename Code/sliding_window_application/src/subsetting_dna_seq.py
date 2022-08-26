import pandas as pd
import os
import torch
import sys

# # DNA_SEQ="C:/Users/uzunv/repos/sampling_chromosome/raw_data/chr12_seq_25step.txt"
# # N=1000000
# # SUB_FILE="_dna_seq_25step.txt"
# # OUTPUT_DIR="C:/Users/uzunv/repos/sampling_chromosome/by100k_step25_chr21"

def splitting_file_into_chunks(DNA_SEQ, N, OUTPUT_DIR, SUB_FILE):
    i=0
    measure_string="chunk_by"+str(N)
    if N==100000:
        measure_string="00k_by100k"
    if N==1000000:
        measure_string="m_by1mil"
    with open(DNA_SEQ, "r") as seqfile:
        while True:
            i+=1
            print(i)
            topn=[]
            try:
                for _ in range(N):
                    topn.append(next(seqfile))
            except StopIteration:
                pass
            #topn = [next(seqfile) for x in range(N)] # this needs fixing
            outfile_name=OUTPUT_DIR+"/till"+str(int(i))+measure_string+SUB_FILE
            with open(outfile_name, "w") as subfile:
                subfile.writelines(topn)
            if len(topn)<N:
                print("All done!")
                break
    

if __name__=="__main__":
    total_dna_seq_subseqed=sys.argv[1]
    chunk_size=int(sys.argv[2])
    outputdir=sys.argv[3]
    subseq_step=sys.argv[4]
    
    suffix="_dna_seq_"+str(subseq_step)+"step.txt"
    
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    
    splitting_file_into_chunks(total_dna_seq_subseqed, chunk_size, outputdir, suffix)