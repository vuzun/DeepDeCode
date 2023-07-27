import itertools
import random
import os
import sys
import numpy as np
import pandas as pd
import time
import datetime
import json
import pathlib
import argparse
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import get_context
import pickle

from models import *
from metrics import *
from train_utils import *
from train import *

from model_hyperparameters_for_search import *

curr_dir_path = str(pathlib.Path().absolute())

NO_MODELS = 64 #32 #4 #512 #16
FILE_NAME = 'hyperparameter_opt_LSTM_classification.csv'
MULTIPROCESS = True
TEST=False

# pickle dictionary for logging hyperparams
HP_DICT_LIST_FILE="SimpleLSTM_hp_dict.pkl" 
model_2_pkl_file_name = {
    "AttLSTM": "AttLSTM_hp_dict.pkl",
    "SimpleLSTM": "SimpleLSTM_hp_dict.pkl",
    "CNN": "CNN_hp_dict.pkl"
}

def helper_func(func_args):
    obj=func_args[0]
    hyperparam_dict=func_args[1]
    obj.training_pipeline()
    return obj.best_metrics, hyperparam_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing different hyperparameters for a model')
    parser.add_argument('--config', type = str, help='config.json with model specifcations', required=True)
    parser.add_argument('--params_file', type = str, help='Parameters file with path locations (system_specific_params.yaml)', required=True)
    parser.add_argument('--output', type = str, help='Output .csv name')
    args = parser.parse_args()

    if args.config==None or args.params_file==None:
        print("Missing args!")
        sys.exit()
    
    if args.output!=None:
        FILE_NAME=args.output

    print(f"Config file: {args.config}, output .csv: {FILE_NAME}")
    print(f"Starting at {str(datetime.datetime.now())}")

    # Get config file
    with open(args.config, encoding='utf-8') as json_data: # , errors='ignore'
        config = json.load(json_data, strict=False)
    #Get System-specific Params file
    with open(args.params_file, 'r') as params_file:
        sys_params = yaml.safe_load(params_file)

    if TEST:
        run_type="test/"
    else:
        run_type="all/"

    # Setting paths
    # "test"/"all"
    final_data_path = sys_params['DATA_WRITE_FOLDER'] + '/' +  run_type + config['DATASET_TYPE'] + \
                      config["DATA"]["DATA_DIR"]
                      
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('_%d-%m_%H:%M')
    model_name_save_dir = string_metadata(config) + timestamp

    save_dir_path = sys_params['HYPER_BASE_FOLDER'] + '/saved_models/' + config[
        'DATASET_TYPE'] + '/' + config["DATA"]["DATA_DIR"] + '/' + model_name_save_dir
        
    tb_path = sys_params['HYPER_BASE_FOLDER'] + '/runs/' + config['DATASET_TYPE'] + \
              '/' + config["DATA"]["DATA_DIR"] + '/' + model_name_save_dir
              
    att_path = sys_params['HYPER_BASE_FOLDER'] + '/attention/' + config['DATASET_TYPE'] + \
              '/' + config["DATA"]["DATA_DIR"] + '/' + model_name_save_dir



    # CNN/SimpleLSTM/AttLSTM
    model_type=config["MODEL_NAME"]
    HP_DICT_LIST_FILE = model_2_pkl_file_name[model_type]
    list_hyperparam = model_2_hp_list_dict[model_type]

    cartesian_prod = list(itertools.product(*list_hyperparam))
    #random.shuffle(cartesian_prod)

    all_hp_dicts=[]
    if os.path.exists(HP_DICT_LIST_FILE):
        with open(HP_DICT_LIST_FILE, "rb") as fpkl:
            all_hp_dicts=pickle.load(fpkl)        

    # removing hparameter combinations for ones already done
    hp_values=[tuple(d.values()) for d in all_hp_dicts]
    cartesian_prod=[t for t in cartesian_prod if t not in hp_values]

    # For multi-processing on multiple GPUs
    num_devices = torch.cuda.device_count()
    devices = list(range(num_devices))
    device_count = 0
    func_args = []
    
    print(f"Number of cuda devices: {num_devices}")
    print(f"List of devices: {devices}")

    # Standard metrics + model specific hyperparameters
    hyperparameter_df_columns=['Model Name'] + model_2_var_dict[model_type] + \
        ['Val Loss', 'Val Accuracy', 'Val F1', 'Val Prec', 'Val Recall',
        'Train Loss', 'Train Accuracy', 'Train F1', 'Train Prec', 'Train Recall',
        'Best Epoch']
    hyperparameter_df = pd.DataFrame(columns = hyperparameter_df_columns) 
    full_hyperparam_file_write_path = sys_params['HYPER_BASE_FOLDER'] + '/' + FILE_NAME
    hyperparameter_df.to_csv(full_hyperparam_file_write_path, mode='a', header=True, index=False)

    if TEST:
        config['TRAINER']['epochs']=2


    for i in range(0,NO_MODELS):
        print('********************************')
        print(f'Building Model {i}...')
        print('********************************')


        hyperparams = cartesian_prod[i]

        hp_names=model_2_var_dict[model_type]

        assert(len(hyperparams)==len(hp_names))

        # Slotting hyperparameters into proper config place and making a dict
        hyperparam_dict={}
        for h_ind, h_name in enumerate(hp_names):
            hyperparam_dict[h_name]=hyperparams[h_ind]
            if h_name=='lr':
                config['OPTIMIZER']['lr']=hyperparams[h_ind]
            elif h_name=='dropout':
                config['TRAINER']['droupout']=hyperparams[h_ind]
            elif h_name=='BATCH_SIZE':
                config['DATA'][h_name]=hyperparams[h_ind]
            else:
                config['MODEL'][h_name]=hyperparams[h_ind]
            


        config['TRAINER']['save_dir'] = save_dir_path
        config['TRAINER']['tb_path'] = tb_path

        #print(hyperparam_dict)
        
        # For multi-processing
        if MULTIPROCESS:

            if device_count < num_devices:
                device = 'cuda:' + str(devices[i % num_devices])
                print('Device:', device)
                device_tag="_device-"+str(device)
                
                try:
                    training_object = Training(config, model_name_save_dir+device_tag,
                        final_data_path, save_dir_path+device_tag, tb_path+device_tag, att_path+device_tag,
                        device) 
                except RuntimeError: # in case of CNN mismatched kernel
                    all_hp_dicts.append(hyperparam_dict)
                    data_dict = {'Model Name': model_name_save_dir,
                         **hyperparam_dict,
                         'Val Loss': -1, 'Val Accuracy': -1,
                         'Val F1': -1, 'Val Prec': -1, 'Val Recall': -1,
                        'Train Loss': -1, 'Train Accuracy': -1,
                         'Train F1': -1, 'Train Prec': -1, 'Train Recall': -1,
                         'Best Epoch': -1}
                    pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False, index=False)
                    continue
                func_args.append((training_object, hyperparam_dict))
                device_count += 1

            if device_count == num_devices or i + 1 == NO_MODELS:
                
                context=get_context('spawn') # default is 'fork' and doesn't work with CUDA on EI/NBI SLURM
                with Pool(num_devices, mp_context=context) as pool:

                    # Train model
                    print('Start Training Model: {}...'.format(model_name_save_dir))
                    pool_result = pool.map(helper_func, func_args) # (metrics, hp_dict) per model

                # Write hyper-parameters and results to csv file
                print(f'Writing hyper-parameters and results file "{full_hyperparam_file_write_path}"')
                
                for metrics, hp_dict in pool_result:
                    train_met = metrics['train'] 
                    val_met = metrics['val']
                    
                    data_dict = {'Model Name': model_name_save_dir,
                                 **hp_dict, #'Batch Size': batch_size, 'Hidden dim': hd, 'No. Layers': nl, 'LR': learning_rate, 'Dropout': dropout,
                             'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                             'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                             'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                             'Train F1': train_met['f1'], 'Train Prec': train_met['prec'],
                             'Train Recall': train_met['recall'],
                             'Best Epoch': train_met['best_epoch']}
                    pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False, index=False)
                    
                    all_hp_dicts.append(hp_dict)

                # Pool over devices finished, resetting for next
                func_args = []
                device_count = 0

        else:
            # Single GPU-mode
            print("Single GPU mode")
            print('Start Training Model: {}...'.format(model_name_save_dir))
            obj = Training(config, model_name_save_dir, final_data_path, save_dir_path, tb_path)
            obj.training_pipeline()

            #Write hyper-parameters and results to csv file
            print('Writing hyper-parameters and results file "{}"'.format(full_hyperparam_file_write_path))
            train_met = obj.best_metrics['train']
            val_met = obj.best_metrics['val']
            
            data_dict = {'Model Name': model_name_save_dir,
                         **hyperparam_dict,
                         'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                         'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                        'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                         'Train F1': train_met['f1'], 'Train Prec': train_met['prec'], 'Train Recall': train_met['recall'],
                         'Best Epoch': train_met['best_epoch']}
            pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False, index=False)

    # saving all hyperparameter dictionaries for future checks
    with open(HP_DICT_LIST_FILE, "wb") as fpkl:
        pickle.dump(all_hp_dicts, fpkl)