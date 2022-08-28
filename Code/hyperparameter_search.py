import tqdm
import itertools
import random
import os
import sys
import shutil
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

from models import *
from metrics import *
from train_utils import *
from train_attention import *

curr_dir_path = str(pathlib.Path().absolute())

NO_MODELS = 4 #512
FILE_NAME = 'boundary60_orNot_2classification.csv'
MULTIPROCESS = True
TEST=False

def helper_func(obj):
    obj.training_pipeline()
    return obj.best_metrics

if __name__ == '__main__':

    # >>
    # pass config file / sys_params file +
    # generalise model type and params (from config)
    # name for save file
    
    
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

    # Set paths
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


    # parameters for DDC model (LSTM params)
    h_batch_size = [32, 64, 128, 256]
    h_hidden_dims = [8, 16, 32, 64, 128]
    h_num_layers = [1, 2, 3]
    h_lr = [0.1, 0.01, 0.001, 0.0001]
    h_dropout = [0.0, 0.3, 0.5]
    
    list_hyperparam = [h_batch_size, h_hidden_dims, h_num_layers, h_lr, h_dropout]
    cartesian_prod = list(itertools.product(*list_hyperparam))
    random.shuffle(cartesian_prod)

    # For multi-processing on multiple GPUs
    #devices=list(np.repeat([0,1,2,4,5,6,7,8,9], 2))  # 9 GPUS; each GPU can have 2 models # ???
    devices = [d for d in range(torch.cuda.device_count())]
    num_devices = len(devices)
    device_count = 0
    func_args = []
    
    print(f"Number of cuda devices: {torch.cuda.device_count()}")

    hyperparameter_df = pd.DataFrame(columns = ['Model Name','Batch Size','Hidden dim','No. Layers', 'LR', 'Dropout',
                                                'Val Loss', 'Val Accuracy', 'Val F1', 'Val Prec', 'Val Recall',
                                                'Train Loss', 'Train Accuracy', 'Train F1', 'Train Prec', 'Train Recall',
                                                'Best Epoch'])
    full_hyperparam_file_write_path = sys_params['HYPER_BASE_FOLDER'] + '/' + FILE_NAME
    hyperparameter_df.to_csv(full_hyperparam_file_write_path, mode='a', header=True) # mode='a'


    for i in range(0,NO_MODELS):
        print('********************************')
        print('Building Model {} ...'.format(i))
        print('********************************')
        hyperparams = cartesian_prod[i]
        batch_size = hyperparams[0]
        hd = hyperparams[1]
        nl = hyperparams[2]
        learning_rate = hyperparams[3]
        dropout = hyperparams[4]

        #Create required config file
        config['DATA']['BATCH_SIZE'] = batch_size
        
        
        # hardcode test
        #hd=16
        #nl=3
        
        config['MODEL']['hidden_dim'] = hd
        config['MODEL']['hidden_layers'] = nl
        
        config['OPTIMIZER']['lr'] = learning_rate
        config['TRAINER']['dropout'] = dropout

        config['TRAINER']['save_dir'] = save_dir_path
        config['TRAINER']['tb_path'] = tb_path

        print(f"Hidden dim: {hd}, hidden layers: {nl}, dropout : {dropout}")
        
        # For multi-processing
        if MULTIPROCESS:

            if device_count < num_devices:
                device = 'cuda:' + str(devices[i % num_devices])
                print('DEVICE:', device)
                device_tag="_device-"+str(device)
                print(device_tag)
                #randomtag = "_"+str(randint(1,9999))
                training_object = Training(config, model_name_save_dir+device_tag,
                    final_data_path, save_dir_path+device_tag, tb_path+device_tag, att_path+device_tag,
                    device)
                func_args.append(training_object)
                device_count += 1

            if device_count == num_devices or i + 1 == NO_MODELS:
                
                context=get_context('spawn') # default is 'fork' and doesn't work with CUDA on NBI SLURM
                with Pool(num_devices, mp_context=context) as pool:

                    # Train model
                    print('Start Training Model: {}...'.format(model_name_save_dir))
                    best_mets = pool.map(helper_func, func_args, chunksize=1)

                best_metrics_list = list(best_mets)

                # Write hyper-parameters and results to csv file
                print('Writing hyper-parameters and results file "{}"'.format(full_hyperparam_file_write_path))
                for met in best_metrics_list:
                    train_met = met['train'] # a mess
                    val_met = met['val']

                    data_dict = {'Model Name': model_name_save_dir, 'Batch Size': batch_size, 'Hidden dim': hd, 'No. Layers': nl,
                             'LR': learning_rate, 'Dropout': dropout, 'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                             'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                             'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                             'Train F1': train_met['f1'], 'Train Prec': train_met['prec'],
                             'Train Recall': train_met['recall'],
                             'Best Epoch': train_met['best_epoch']}
                    pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False)

                func_args = []
                device_count = 0

        else:
            # Single GPU-mode
            # Train model
            print("Single GPU mode")
            print('Start Training Model: {}...'.format(model_name_save_dir))
            obj = Training(config, model_name_save_dir, final_data_path, save_dir_path, tb_path)
            obj.training_pipeline()

            #Write hyper-parameters and results to csv file
            print('Writing hyper-parameters and results file "{}"'.format(full_hyperparam_file_write_path))
            train_met = obj.best_metrics['train']
            val_met = obj.best_metrics['val']
            data_dict = {'Model Name': model_name_save_dir, 'Batch Size': batch_size, 'Hidden dim': hd, 'No. Layers': nl,
                         'LR':learning_rate, 'Dropout':dropout, 'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                         'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                        'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                         'Train F1': train_met['f1'], 'Train Prec': train_met['prec'], 'Train Recall': train_met['recall'],
                         'Best Epoch': train_met['best_epoch']}
            pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False)
