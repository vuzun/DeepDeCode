# %%
import pandas as pd
import numpy as np
import os
import torch
import sys

# N_CHUNK=2
# BY_SIZE=100000
# STEP=25
# SEQ_LENGTH=100
# SEQ_SCORE="output_chr19_till"+str(N_CHUNK)+"00k_by100k_25s.pt"
# SIMPLIFIED_GTF="annotation/gencode_v38_novel_only_exon_start_end.tsv"
# CHR_NUM=19

# %%
# from results
def generate_scores_df_for_chromosome_chunk(N_CHUNK, BY_SIZE, STEP, SEQ_LENGTH,
    SEQ_SCORE, SIMPLIFIED_GTF, CHR_NUM):
    """
    Makes a pandas data frame with scores, chr positions and known junction class for sequences.
    Using DeepDeCode model's output, a file with exon start/ends and size arguments.
    """    

    scores=torch.load(SEQ_SCORE)
    
    # attention model saved a tuple, unlike the rest
    if type(scores)==tuple:
        scores=scores[0]
    #scores=torch.load(SEQ_SCORE)[0] # why 0 here? V23
    df=pd.DataFrame(scores.detach().numpy())
    df["class"]=0
    df["line"]=df.index.values+1
    df.columns.values[:2]=["score","score2"]

    # reduced exon edges from GTF
    exons=pd.read_csv(SIMPLIFIED_GTF, sep="\t", header=None)

    num_bases_before=(N_CHUNK-1)*BY_SIZE*STEP
    num_last_base=num_bases_before+BY_SIZE*STEP

    df_start=num_bases_before+STEP*(df["line"]-1)+1
    df["start"]=df_start
    df["end"]=df_start+SEQ_LENGTH-1


    EXON_START_SIGN=1
    EXON_END_SIGN=2
    junction_list=[]
    for i in range(int(exons.shape[0])):
        # check if num_last_base or before
        exon_start=exons.iloc[i,0]
        exon_end=exons.iloc[i,1]

        if num_bases_before > exon_end or num_last_base < exon_start:
            continue 
        
        # for the first ocurrence
        # line start would be num_bases_before+STEP*some_row_index + a_bit
        #row_index_start=int((exon_start-num_bases_before)//STEP)
        #row_index_end=int((exon_end-num_bases_before)//STEP)
        
        # >change to ones that match in perfectly at +50 from start
        # subsequences that contain the junctions (anywhere)
        lines_exon_start=df[(df["start"]<=exon_start) & (df["end"]>=exon_start)].index.values
        lines_exon_end=df[(df["start"]<=exon_end) & (df["end"]>=exon_end)].index.values

        # because the sequences are overlapping, a single position in the real sequence
        # will appear several times here
        # for SEQ_LENGTH/STEP times with this exon_start
        #for i in range(int(SEQ_LENGTH/STEP)):

        # exon start update
        # updates all subsequences which contain the exon anywhere!
        for row_index in lines_exon_start:
            if df.iloc[row_index, df.columns.get_loc("start")]+49!=exon_start:
                continue
            if num_bases_before < exon_start:

                current_class=df.iloc[row_index, df.columns.get_loc('class')]
                seq_start_estart=df.iloc[row_index, df.columns.get_loc("start")]
                
                # if current class gets too big with many start/ends, it will cause error
                if current_class>10**11:
                    continue
                
                if current_class==0:
                    new_class=EXON_START_SIGN
                else:
                    # concatenate, so it's visible if both start and end, or multiple starts
                    new_class=int(str(current_class)+str(EXON_START_SIGN))
            
                df.iloc[row_index, df.columns.get_loc('class')]=new_class  

        # exon end update
        for row_index in lines_exon_end:
            if df.iloc[row_index, df.columns.get_loc("start")]+49!=exon_end:
                continue        
            if num_last_base > exon_end:
                current_class=df.iloc[row_index, df.columns.get_loc('class')]
                seq_start_eend=df.iloc[row_index, df.columns.get_loc("start")]
                
                # if current class gets too big with many start/ends, it will cause error
                if current_class>10**11:
                        continue

                if current_class==0:
                    new_class=EXON_END_SIGN
                else:
                    # concatenate, so it's visible if both start and end, or multiple starts
                    new_class=int(str(current_class)+str(EXON_END_SIGN))

                # df.iloc[row_index,]["class"]=new_class # this doesn't work because pandas is senseless
                df.iloc[row_index,df.columns.get_loc('class')]=new_class

        # ~till here
    return(df)
    

 
# %%
if __name__=="__main__":
    
    n_chunk=int(sys.argv[1])
    seq_score=sys.argv[2]#"results/output_chr"+str(chr_num)+"_till"+str(n_chunk)+"00k_by100k_25s.pt"
    by_size=int(sys.argv[3]) #100000
    step=int(sys.argv[4])#25
    seq_length=int(sys.argv[5])#100
    chr_num=int(sys.argv[6])
    
    # this might need to be an arg
    simplified_gtf="annotation/gencodev38_chr"+str(chr_num)+"_only_exon_start_end_unique.tsv"
    
    #print(len(sys.argv))
    #print(sys.argv)
    folder_name="results/"
    if len(sys.argv)>7:
        folder_name=sys.argv[7]
        if not folder_name.endswith(os.path.sep):
            folder_name += os.path.sep
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        
    if len(sys.argv)>8:
        simplified_gtf=sys.argv[8]

    range_string="N_byNchunk"
    if by_size==100000:
        range_string="00k_by100k_"
    elif by_size==1000000:
        range_string="1m_by1m_"
    elif by_size==10000:
        range_string="0k_by10k_"

    df_filename=folder_name+"scores_df_chr"+str(chr_num)+"_till_"+str(n_chunk)+range_string+str(step)+"s.pt"

    df=generate_scores_df_for_chromosome_chunk(N_CHUNK=n_chunk, BY_SIZE=by_size, STEP=step,SEQ_LENGTH=seq_length,
    SEQ_SCORE=seq_score, SIMPLIFIED_GTF=simplified_gtf, CHR_NUM=chr_num)  

    df.to_pickle(df_filename, protocol=4)

    # df["class"].describe()
    # df["class"].unique()
    # df["class"].value_counts()

    # df[df["class"]!=0]["score"].describe()
    # df["score"].describe() #  for this chunk of chr19, mean .38, meadian .15
