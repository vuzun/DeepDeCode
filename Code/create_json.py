import pathlib
import pandas as pd
from os import path
import json
import itertools
#import Bio
import argparse
import os

curr_dir_path = str(pathlib.Path().absolute())
raw_data_path = curr_dir_path + "/raw_data/"

def gene_start_end_positions(chr_ann):
    '''
    Function to read annotation for given chromosome and get [start, end] base pair position of genes
    :param chr_ann: data frame
            Annotations for given chromosome
    :return: ndarray (num_genes, 2)
            Array containing [start, end] base pair position on chromosome for each gene.
    '''
    chr_gene = chr_ann[chr_ann['type'] == 'gene']
    gene_start_end_pos = pd.DataFrame(
        {'start_pos': chr_gene['start'], 'end_pos': chr_gene['end']})  # shape: (875,2) for chr21
    gene_start_end_pos.reset_index(drop=True, inplace=True)
    print('Start,end shape:', gene_start_end_pos.shape)
    return gene_start_end_pos

def get_indices_of_table(df, last_index):
    '''
    Get indices of the rows of the (sliced) data frame [which, in this case df corresponds to 'type' gene]
    :param df: data frame
    :return: list of indexes of the data frame
    '''
    indices = pd.Index.to_list(df.index)
    indices.append(last_index)

    return indices

def no_transcripts_per_gene(cur, nex, chr_ann):
    '''
    Function to get number of transcripts between the cur and nex index of the chr_ann data frame)
    :param cur: int: current index
    :param nex: int: next index
    :param chr_ann: data_frame: annotation file for a particular chromosome
    :return: int: no. of transcripts found
    '''
    gene_table = chr_ann.iloc[cur:nex]
    transcript_counts_per_gene=gene_table['type'].value_counts()['transcript']

    return transcript_counts_per_gene

def get_chunk(cur, nex, chr_ann, type):
    '''
    Function to get subset of the annotation data frame according to condition
    :param cur: int: Start index
    :param nex: int: End index
    :param chr_ann: data frame: Annotation data frame
    :return: data frame
            Subset of data frame
    '''
    table = chr_ann.iloc[cur:nex]
    df = table[table['type'] == type]
    return df

def write_meta_data(chr_no, chr_dictionary, chr_seq):
    '''
    Write chromosome meta-info to file (appending)
    :param chr_no: int: chromosome no.
    :param chr_dictionary: dict: dictionary containing the relevant chromosome info
    :param chr_seq: str: Nucleotide seq for the chromosome
    :return: None
    '''
    print('Writing meta-info to log file...')
    log_file = open(raw_data_path + 'json_files/length_info.log', "a+")
    log_file.write('\n\nCHROMOSOME {} ---------'.format(chr_no))
    log_file.write('\nNo. of genes in chromosome: {}'.format(len(chr_dictionary['main'])))
    log_file.write('\nLength of chromosome sequence : {} nts.'.format(len(chr_seq)))
    log_file.close()

def create_dict(chr_seq, chr_ann, transcript_element="CDS"):
    '''
    Annotation/sequence dictionary for a chromosome. {'main': [{'gene_id':, 'gene_strand':,...},{...},...]}
    :param chr_seq: string
            Nucleotide sequence of the chromosome
    :param chr_ann: data frame
            Annotation file for the chromosome
    :param transcript_element: str
            Which subtranscript element to gather
            'CDS' or 'exon'
    :return: dictionary
        Dictionary to be stored as the json file
    '''
    chr_gene = chr_ann[chr_ann['type'] == 'gene'] 

    gene_ids = list(chr_gene['0'])  # '0' is first column of the rest of the .gtf, containing gene id

    gene_start_end_pos = gene_start_end_positions(chr_ann)
    gene_bounds = list(zip(gene_start_end_pos.start_pos, gene_start_end_pos.end_pos))

    gene_strand = list(chr_gene['strand'])

    indices = list(range(0, len(gene_start_end_pos)))
    gene_sequence = list(map(lambda x: chr_seq[gene_bounds[x][0]:gene_bounds[x][1]],
                             indices))
    
    gene_indices = get_indices_of_table(chr_gene, len(chr_ann)) # gene indices the way they were in chr_ann

    chr_dictionary = {'main': []}

    for i in range(0,len(chr_gene)):
        cur = gene_indices[i]
        nex = gene_indices[i+1]
        gene_dict = {}

        gene_dict.update({'gene_id': gene_ids[i]})
        gene_dict.update({'gene_strand': gene_strand[i]})
        gene_dict.update({'gene_bounds': gene_bounds[i]})
        gene_dict.update({'gene_sequence':gene_sequence[i]})  #take into account reverse complementarity

        no_transcripts = no_transcripts_per_gene(cur, nex, chr_ann)
        gene_dict.update({'no_of_transcripts': int(no_transcripts)})

        transcripts = []
        transcript = get_chunk(cur, nex, chr_ann, 'transcript')
        transcript_ids = [l.strip('"') for l in list(transcript['1'])]
        transcript_ranges = list(zip(transcript['start'], transcript['end']))
        transcript_indices = get_indices_of_table(transcript, nex)

        for j in range(0,len(transcript_ranges)):

            transcript_dict = {}

            transcript_dict.update({'transcript_id': transcript_ids[j]})
            transcript_dict.update({'transcript_range': transcript_ranges[j]})

            if transcript_element=="CDS":
                cds = get_chunk(transcript_indices[j], transcript_indices[j+1], chr_ann, transcript_element)
                cds_ranges = list(zip(cds['start'], cds['end']))
                cds = []

                for bound in cds_ranges:
                    cds.append({'cds_ranges': bound})

                transcript_dict.update({'no_of_cds': int(len(cds_ranges))})
                transcript_dict.update({'cds': cds})

            elif transcript_element=="exon":
                exons = get_chunk(transcript_indices[j], transcript_indices[j+1], chr_ann, transcript_element)
                exon_ranges = list(zip(exons['start'], exons['end']))
                exons = []

                for bound in exon_ranges:
                    exons.append({'exon_ranges': bound})

                transcript_dict.update({'no_of_exons': int(len(exon_ranges))})
                transcript_dict.update({'exons': exons})
                
            else:
                raise ValueError("Invalid value for transcript_element!")

            transcripts.append(transcript_dict)

        gene_dict.update({'transcripts': transcripts})
        chr_dictionary['main'].append(gene_dict)

    return chr_dictionary

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create individual cds json files')
    parser.add_argument('--no', type = str, help='chromosome number')
    parser.add_argument('--exon', type = str, help='if exons should be gathered per transcript, as opposed to cds')
    args = parser.parse_args()

    transcript_element="CDS"
    if args.exon!=None:
        transcript_element="exon"
        print("Exon selected")

    print('Starting JSON processing Chromosome {}'.format(args.no))
    chr_txt_file = "chr{}.txt".format(args.no)
    chr_ann_file = "chr{}_annotations.csv".format(args.no)


    with open(raw_data_path + 'txt/' + chr_txt_file, "r") as file_object:
        chr_seq = file_object.read()

    chr_ann = pd.read_csv(raw_data_path + 'annotations/' + chr_ann_file, sep='\t')

    chr_dictionary = create_dict(chr_seq, chr_ann, transcript_element)

    if not os.path.exists(raw_data_path+'json_files'):
        os.makedirs(raw_data_path+'json_files')
    print('Writing to json file...')

    with open(raw_data_path+'json_files/'+'chr'+str(args.no)+'_cds_data.json', 'w') as file:
        json.dump(chr_dictionary, file)

    write_meta_data(args.no, chr_dictionary, chr_seq)
    print('Finished JSON processing Chromosome {}'.format(args.no))