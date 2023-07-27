# %%
import sys, os
import numpy as np
import pandas as pd
import re
from time import time

# %%
# gets all files (scores_df) from a folder and then 
# extracts class stats for all and returns >threshold and <threshold df summaries
def extract_stats_from_scores_df(path_to_dfs, threshold=0.8):
    
    print(f"Extracting stats for threshold {threshold}!")
    print(f"From folder: {path_to_dfs}")

    filenames_score_dfs=os.listdir(path_to_dfs)
    print(f"{len(filenames_score_dfs)} files found!")

    stats_above_threshold=pd.DataFrame()
    stats_under_threshold=pd.DataFrame()  
    i=0
    for scores_df in filenames_score_dfs:
        chunk_name=re.search("_[0-9]+k_", scores_df).group()[1:-1] # naming counts by chunk number
        #print(chunk_name)
        i+=1
        if i%10==0:
            print(i)
        
        # adding / or \ if not already there for name creation later
        if not path_to_dfs.endswith(os.path.sep):
                path_to_dfs+=os.path.sep
        
        df=pd.read_pickle(path_to_dfs+scores_df)

        stats_series=df[df["score"]>threshold]["class"].value_counts()
        stats_series.name=chunk_name
        stats_above_threshold=pd.concat([stats_above_threshold, stats_series], axis=1)

        stats_neg=df[df["score"]<threshold]["class"].value_counts()
        stats_neg.name=stats_series.name
        stats_under_threshold=pd.concat([stats_under_threshold, stats_neg], axis=1)

    stats_above_threshold=stats_above_threshold.fillna(0).astype(int)
    stats_under_threshold=stats_under_threshold.fillna(0).astype(int)

    return stats_above_threshold, stats_under_threshold



if __name__=="__main__":
    
    threshold=0.8
    output_directory="./"

    directory_with_scores_df=sys.argv[1]

    if len(sys.argv)>2:
        threshold=float(sys.argv[2])

    if len(sys.argv)>3:
        output_directory=sys.argv[3]
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    stats_above, stats_under = extract_stats_from_scores_df(directory_with_scores_df, threshold)

    summary_above=pd.DataFrame(stats_above.sum(axis=1))
    summary_under=pd.DataFrame(stats_under.sum(axis=1))
    summary_above["classes"]=summary_above.index
    summary_under["classes"]=summary_under.index
    summary_above.columns=["over_threshold", "classes"]
    summary_under.columns=["under_threshold", "classes"]

    summary_stats=summary_above.merge(summary_under, on="classes")

    if not output_directory.endswith(os.path.sep):
        output_directory += os.path.sep

    file_name_summary_stats=output_directory+"summary_stats_for_"+str(threshold)+".tsv"
    file_name_stats_above=output_directory+"stats_above_df_for_"+str(threshold)+".pkl"
    file_name_stats_under=output_directory+"stats_under_df_for_"+str(threshold)+".pkl"


    stats_above.to_pickle(file_name_stats_above, protocol=4)
    stats_under.to_pickle(file_name_stats_under, protocol=4)
    summary_stats.to_csv(file_name_summary_stats, sep="\t", index=False)