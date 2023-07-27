# DeepDeCode: Deep Learning for Identifying Important Motifs in DNA Sequences

This is the code base for the DeepDeCode project developed by Vladimir Uzun and Asmita Poddar. The code has been mainly written in Python and the deep learning framework used is PyTorch.


* Main goal of the project was exploring prediction models for splice site junctions (intron/exon boundaries) using DNA sequence and annotation alone
* A combination of LSTM and attention architecture showed best results when compared to standard models
* By leaving out one of the chromosomes and applying a sliding window application method, we obtain scores for each of the bases along a chromosome


## Requirements
- Python >= 3.6   
- PyTorch >= 1.5.0  
- tensorboard >= 1.14 (see Tensorboard Visualization)  
- argparse >= 1.1  
- biopython >=1.76  
- json >= 2.0.9  
- logomaker >= 0.8  
- matplotlib >= 3.1.0  
- numpy >= 1.18.5  
- pandas >= 0.24.2  
- seaboarn >= 0.10.1  
- sklearn >= 0.21.3  
- yaml >= 5.1.2  

## Features
- `.json` file for convenient parameter tuning
- `.yml` file for base path specification to data directory, model directory and visualization(Tensorboard and Attention) directory
- Writing and visualization of model training logs using Tensorboard  
- Using Multiple GPUs for hyper-parameter search

## Data

The human DNA sequence data from the Dec. 2013 assembly of the human genome ([hg38, GRCh38 Genome Reference Consortium Human Reference 38](http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/)) was obtained from the [UCSC Genome Browser](https://genome.ucsc.edu/) [80] in the FASTA format (a text-based format for representing nucleotide sequences, in which nucleotides are represented
using single-letter codes) for each chromosome.   
We obtained the location of exons within the DNA sequences from the latest release ([Release 34, GRCh38.p13](https://www.gencodegenes.org/human/)) of GENCODE annotations [81] in the Gene Transfer Format (GTF), which contains comprehensive gene annotations on the reference chromosomes in the human genome.

## Pre-processing Steps
The aim of pre-processing is to extract the relevant parts of the genome to get high quality data for our models. The following pre-processing steps were performed to create the datasets:

### Scripts   
- [read_annotations.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/read_annotations.py): Read annotation file for human genome sequence GRCh38 and get annotations for specified chromosome
- [create_json.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/create_json_cds.py): To create a JSON file (containing the keys: `gene_id`, `gene_sequence`, `gene_strand`, `no_of_transcript`, `transcript_id`, `transcript_range`, `no_of_exons`, `exon_ranges`) containing infomation about the _Coding DNA Sequences (CDS)_ sequences from the GENCODE annotation file. Can be done based on CDS or exon sequences.
- [dataset_utils.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/dataset_utils.py): Contains the utility functions for creating the positive and negative sequences for our dataset.   
- [process.sh](https://github.com/vuzun/DeepDeCode/blob/master/Code/process.sh): Shell script written to automate the process of creating the dataset and encoding it for all the chromosomes over the full genome. The pipeline consists of creating the text and a JSON files for the indivisual chromosomes, the DNA sequences along with the corresponding labels, and then creating a one-hot encoding for the data.  

### Usage
1. Download the GTF and FASTA files for each of the Chromosomes into the `Data` folder.  
2. Run `python3 create_json_cds.py --no <chromosome_no>` to extract relevant information in the JSON format.
3. Run `./process.sh` to get DNA sequences in the valid regions from the entire genome.

## Dataset Creation
The final dataset for the experiments were created after pre-processing the data. Standard data format used among all my datasets:  
**Input**: The input to the models consist of a DNA sequence of a particluar length, _L_. A single encoded training sample has the dimensions [_L, 4_].  
**Label**: The label (output) of the model depends on the Experiment Type. The label could be:  
- 0 (Contains a splice junction) / 1 (Does not contain a splice junction) : for 2-class classification
- 0 (containing splice junction) / 1 (exon) / 2 (intron) : for 3-class classification
- Any number \[1,_L_\] : for multi-class classification

### Scripts  
- [generate_dataset_types.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/generate_dataset_types.py): Create the datasets tailered for the three experiments (Experiment I: boundaryCertainPoint_orNot_2classification, Experiment II: boundary_exon_intron_3classification and Experiment III: find_boundary_Nclassification).   
- [generate_dataset.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/generate_dataset.py): Containing the entire pipeline to generate the required type of dataset from the created JSON file for a particular chromosome.  
- [generate_entire_dataset.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/generate_entire_dataset.py): To generate the required type of dataset for the entire genome by calling `generate_dataset.py` for every chromosome.    
[encode.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/encode.py): Creates the one-hot encoding of the DNA sequences.    
- [meta_info_stitch.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/meta_info_stitch.py): To obtain the meta-information about our dataset and writing it to file.  
- [subset.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/subset.py): Create a random subset of a specified number of samples from a larger dataset. 

### Usage
1. Run `python3 generate_dataset.py` to get single chromosome dataset or `python3 generate_entire_dataset.py` to get multiple chromosome dataset for specified Experiment type.  
2. Run `python3 subset.py -i <file path of dataset> -n <no. of samples in subset>` to create a subset of the dataset (typically used for faster training to create a POC).

## Model Training  
The model architectures that we implemented are:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory Network (LSTM)
- DeepDeCode

### Scripts
- [train.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/train.py): Generalised training pipeline for classification or regression with the different architectures. The training pipeline consists of passing the hyper-parameters for the model, training for each batch, saving the trained models, writing to Tensorboard for visualization of the training process, writing training metrics (loss, performance metrics) to file. 
- [train_utils.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/train_utils.py): Contains utility function for training the deep learning models.  
- [train_Kfold.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/train_Kfold.py): Training with K-fold cross validation to prevent model over-fitting. 
- [models.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/models.py): Contains the various model architectures used for our experiment. The architectures implemented are CNN, LSTM and DeepDecode. 
- [test_model.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/test_model.py):To evaluate the trained models using the test set for classification tasks. 
- [test_regression.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/test_regression.py): To evaluate the trained models using the test set for classification tasks
- [hyperparameter_search.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/hyperparameter_search.py): To perform hyper-parameter search for the specified hyper-paramters for the various models over a search space. the results for each model are stored in a CSV file.  
- [metrics.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/metrics.py): Calculates the evaluation metrics for our models.

### Usage
1. Run `python3 train.py` or `python3 train_attention.py` to train the model without and with attention respectively.
2. Run `python3 hyperparameter_search.py --FILE_NAME <file name to write metrics> --ATTENTION True --devices <GPU numbers to use>` for hyper-parameter searching for the models. For LSTM models, you can either turn the attention mechanism on or off by specifying the value for the `--ATTENTON` parameter.
3. Get test metrics for a model by running `python3 test_model.py`. 

#### Config file format  

Config files are in `.json` format:
```
{
    "EXP_NAME": "Len100",                                            // training session name
    "TASK_TYPE": "classification",                                   // classification/regression
    "DATASET_TYPE": "boundaryCertainPoint_orNot_2classification",    // type of dataset being used for experiment
    "MODEL_NAME": "AttLSTM",                                         // model architecture
    "LOSS": "CrossEntropyLoss",                                      // loss
    "DATA": {
        "DATALOADER": "SequenceDataLoader",           // selecting data loader
        "DATA_DIR": "/end/100",                       // dataset path
        "BATCH_SIZE": 32,                             // batch size
        "SHUFFLE": true,                              // shuffle training data before splitting
        "NUM_WORKERS": 2                              // number of cpu processes to be used for data loading
    },
    "VALIDATION": {
        "apply": true,                    // whether to have a validation split (true/ false)
        "type": "balanced",               // type of validation split (balanced/mixed/separate)
        "val_split": 0.1,                 // size of validation dataset - float (portion of samples)
        "cross_val": 10                   // no. of folds for K-fold cross validation
    },
    "MODEL": {                            // Hyper-parameters for LSTM-based model
        "embedding_dim": 4,                
        "hidden_dim": 32,
        "hidden_layers": 2,
        "output_dim": 2,
        "bidirectional": true
    },
    "OPTIMIZER": {
        "type": "Adam",                   // optimizer type
        "lr": 0.1,                        // learning rate
        "weight_decay": 0.0,              // weight decay
        "amsgrad": true            
    },
    "LR_SCHEDULER": {
        "apply": true,            // whether to apply learning rate scheduling (true/ false)
        "type": "StepLR",         //type of LR scheduling.  More options available at https://pytorch.org/docs/stable/optim.html
        "step_size": 20,          // period of learning rate deca
        "gamma": 0.05             // multiplicative factor of learning rate decay
    },
    "TRAINER": {
        "epochs": 110,                               // number of training epochs
        "dropout": 0.1,                              // dropout
        "save_all_model_to_dir": false,              // whether to save models of every epoch (true/false)
        "save_model_to_dir": true,                   // whether to save any model to directory (true/false)
        "save_dir": "/all/att_start/",               // path for saving model checkpoints: './saved_models/all/att_start'
        "save_period": 250,                          // save checkpoints every save_period epochs
        "monitor": "acc",                            // metric to monitor for early stopping
        "early_stop": 50,                            // number of epochs to wait before early stop. Set 0 to disable
        "tensorboard": true,                         // enable tensorboard visualizations
        "tb_path": "chr21/end_half/Len30_model"      // path for tensoroard visualizations: './runs/chr21/end_half/Len30_model'
    }
}
```
This file is used during model training. The values in the required fields can be changed to set paths or parameters.  

#### Tensorboard Visualization
1. **Run Training**: Make sure that tensorboard option in the `config` file is turned on: `"tensorboard" : true`.  
2. **Open Tensorboard server**: In the command line, type `tensorboard --logdir runs/log/ --port 6006` at the project root, then server will open at `http://localhost:6006`. If you want to run the tensorboard visualizations from a remote server on your local machine using SSH, run the following on your local computer:
```
ssh -N -f -L localhost:16006:localhost:6006 <remote host username>@<remote host IP>
tensorboard --logdir runs --port 6006
```
The server will open at: `http://localhost:6006`  

## Visualizations 
Inference of biologically relevant information learnt by models in the genomic domain is a challenge. We identify sequence motifs in the genome that code for exon location. We explore various intrinsic and extrinsic visualization techniques to find the important sequence motifs informing the the existence of acceptor sites or donor sites.  
     
### Scripts 
   
- [perturbation_test.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/perturbation_test.py): To perform the sequence pertubation test using the trained models. Various lengths of perturbations can be performed over the DNA sequences.  
- [visualize_attention.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/visualize_attention.py): Visualize the attention maps.  
- [graphs.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/graphs.py): Code to generate the graphs (distribution of exon position for Experiment III, variation of model accuracy for over length of the DNA sequence for Experiment I.
    
### Usage
1. Run `python3 visualize_attention.py -f <file to plot visualization for (all/file_name)> -t <train/val folder> -s <start epoch> -e <end epoch>` to get visualizations from attention maps.  
2. Run `python3 perturbation_test.py` for perturbation tests. 
3. Get various graphs by running `python3 graphs.py`


## Sliding window application pipeline

Sliding window application is useful for better evaluation of the method and inspecting genomic locations of interest. If we select a chromosome to be left out of training and validation data (`chrom_ignore` variable in `generate_entire_dataset.py` contains chromosomes to be skipped), we can apply a trained model on all the positions of a chromosome for evaluation in a sequential, or sliding, manner.

By using a sliding subsequence window, we generate all possible inputs from a left out chromosome. This enables application of the trained model on every point of an unseen genomic region.

Unless only a small subset of chromosome is being used, or sliding window has a significant step, it is strongly suggested to use a computer cluster due to memory and time constraints. We have used SLURM HPC. Specific job commans are provided.

### Scripts

- [chr_prep_window.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/sliding_window_application/src/chr_prep_window.py): Uses sequence data (pre-processed with read_annotations.py) to create a single file with lines of subsequence length with "step" difference between them (default:1)
- [subsetting_dna_seq.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/sliding_window_application/src/subsetting_dna_seq.py): Splits the result of previous step into multiple files of a specified chunk size
- [encode.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/encode.py): Same as before, used to encode the chunks.
- [applying_deepdecode_by_chunks.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/sliding_window_application/src/applying_deepdecode_by_chunks.py): With a trained model checkpoint, applies the loaded model to an encoded chunk.
- [scores_chr_position_match.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/sliding_window_application/src/scores_chr_position_match.py): Matches GENCODE splicing boundary annotation onto DeepDeCode scores of positions per chunk
- [extract_stats_from_scores_df.py](https://github.com/vuzun/DeepDeCode/blob/master/Code/sliding_window_application/src/extract_stats_from_scores_df.py): Summaries results of a group of chunks

### Usage
Examples of SLURM jobs used to submit these to the HPC cluster are in [Code/sliding_window_application](Code/sliding_window_application).

1. Using a chromosome file preprocessed with `read_annotations.py`, to generate a format for sublength of 100 and with a window sliding step of 25: `python3 chr_prep_window.py chr1.txt chr1prepped.txt 100 25`
2. The resulting `chr1prepped.txt` is split into many chunks based on the number of lines desired per chink. For 100000 lines and step 25 used to generate chunks in a directory: `python3 subsetting_dna_seq.py chr1prepped.txt 100000 chunks_dir/ 25`.

3. Every chunk is encoded
    * For a single chunk `till100k_by100k_dna_seq_25step.txt`: `python3  encode_hpc.py --write_path chunks_dir/ --dna_seq till100k_by100k_dna_seq_25step.txt --encoded_seq encoded_till100k_by100k_25step.txt`
    * Using SLURM, for all chunks in a folder `chr1_100kchunks_25step/` that have the same suffix to make encoded chunks with the second suffix: `for i in {1..CHUNK_NUMBER}; do sbatch encode_job.sh chr1_100kchunks_25step/ $i 00k_by100k_dna_seq_25step.txt 00k_by100k_25step.txt; done;`
4. Trained model is applied to every encoded chunk:
    * For a single chunk, using a DeepDeCode model from `model_checkpoint.pt`: `python3 applying_deepdecode_by_chunks.py chr1_100kchunks_25step/encoded_till100k_by100k_25step.txt results/output_chr1_till100k_25s_123 model_checkpoint_123.pt`
    * For all encoded chunks, using SLURM: `for i in {1..CHUNK_NUMBER}; do sbatch ddc_apply.sh chr1_100kchunks_25step/encoded_till${i}00k_by100k_1s results/output_chr1_100kchunks_25s_123/chr1_till${i}00k_by100k_25s.pt model_checkpoint_123.pt; done;`
5. Matching annotation to model scores per position:
    * For a single chunk result `python3 scores_chr_position_match.py 1 results/output_chr1_100kchunks_25s_123/chr1_till100k_by100k_25s.pt 100000 25 100 1 results/scores_df annotation/gencodev38_chr12_only_exon_start_end_unique.tsv; done`
    * For all results : `for i in {1..CHUNK_NUMBER}; do sbatch scores_chr_position_match_job.sh ${i} results/output_chr1_t1s_45656952/chr12_till${i}00k_by100k_1s.pt 100000 1 100 12 results/scores_df_chr12only_by100k_1s_reanno annotation/gencodev38_chr12_only_exon_start_end_unique.tsv; done`

6. `python3 extract_stats_from_scores_df.py sdf_dir 0.8 summary_dir/`

## License
This project is licensed under the MIT License. See LICENSE for more details. 
