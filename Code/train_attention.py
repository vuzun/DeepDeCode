import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np

import copy
from tqdm import tqdm
import os
import shutil
import time
import datetime
import json
import pathlib
import yaml #

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.tensorboard as tb

from train_utils import *
from models import *
from metrics import *

from random import randint

curr_dir_path = str(pathlib.Path().absolute())

class Training():
    def __init__(self, config,
                model_name_save_dir, data_path='', save_dir='', tb_path='', att_path='',
                x_train=None, y_train=None, x_val=None, y_val=None, train_from_checkpoint=False,
                device='', start_epoch=0,
                encoded_seq_filename="encoded_seq", y_label_filename="y_label"):
        self.config = config
        if device == '':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:5' changed to 'cuda'
        else:
            self.device = device

        self.data_path = data_path
        self.save_dir = save_dir
        self.tb_path = tb_path
        self.att_path = att_path

        self.start_epoch = start_epoch

        self.encoded_seq = np.loadtxt(data_path + '/' + encoded_seq_filename) # hardcode
        self.y_label_filename=y_label_filename
        
        
        if self.config['MODEL_NAME']=="AttLSTM":
            no_timesteps = int(len(self.encoded_seq[0]) / 4)
            att_DNA_args = {'n_nts': config['MODEL']['embedding_dim'], 'n_bins': no_timesteps, # 4 x 100
                               'hidden_size': config['MODEL']['hidden_dim'], 'num_layers': config['MODEL']['hidden_layers'],
                               'dropout': config['TRAINER']['dropout'], 'bidirectional': config['MODEL']['bidirectional']}
            self.model = att_DNA(att_DNA_args, config['MODEL']['output_dim'])

        if self.config['MODEL_NAME']=='SimpleLSTM':
            self.model = eval(config['MODEL_NAME'])(config['MODEL']['embedding_dim'], config['MODEL']['hidden_dim'],
                                                config['MODEL']['hidden_layers'], config['MODEL']['output_dim'],
                                                config['DATA']['BATCH_SIZE'], config['MODEL']['bidirectional'],
                                                config['TRAINER']['dropout'], self.device)
        if self.config['MODEL_NAME']=='CNN':
            self.model = eval(config['MODEL_NAME'])(config['MODEL']['output_dim'], self.device)

        self.optimizer = getattr(optim, config['OPTIMIZER']['type']) \
            (self.model.parameters(), lr=config['OPTIMIZER']['lr'], weight_decay=config['OPTIMIZER']['weight_decay'])
        self.scheduler = getattr(optim.lr_scheduler, config['LR_SCHEDULER']['type']) \
            (self.optimizer, step_size=config['LR_SCHEDULER']['step_size'],
             gamma=self.config['LR_SCHEDULER']['gamma'])

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.trainloader = None
        self.writer = {'train': None, 'val': None}  # For TensorBoard

        self.metrics = {'train': {}, 'val': {}}
        self.best_metrics = {'train': {}, 'val': {}}
        self.model_name_save_dir = model_name_save_dir
        self.train_from_checkpoint=train_from_checkpoint

    def save_checkpoint(self, epoch, save_type=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'checkpoint_best.pth'
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if save_type=='any':
            filename = self.save_dir + str('/checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))
        if save_type=='best':
            best_path = str(self.save_dir + '/best_checkpoint{}.pth'.format(epoch))
            torch.save(state, best_path)
            print("Saving current best: best_checkpoint.pth ...")
        if save_type=='last':
            last_path = str(self.save_dir + '/last_checkpoint{}.pth'.format(epoch))
            torch.save(state, last_path)
            print("Saving the last: last_checkpoint.pth ...")

    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        if not os.path.isfile(resume_path):
            raise Exception("Failed to read path %s, aborting." % resume_path)
            return

        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        #self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        assert checkpoint['config']['MODEL_NAME'] != self.config['MODEL_NAME'], \
            "Warning: Architecture configuration given in config file is different from that of checkpoint. "
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['OPTIMIZER']['type'] != self.config['OPTIMIZER']['type']:
            print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {} ------".format(self.start_epoch))

    def write_model_meta_data(self, x_train, x_val, y_train, y_val,
                              train_idx, val_idx):  # todo might move to train_utils later
        '''
        Write meta-info about model to file
        '''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with open(self.save_dir+'/config.json', 'w') as outfile1:
            json.dump(self.config, outfile1, indent = 4)

        log_file = open(self.save_dir + '/info.log', "w+")
        log_file.write(str(self.model))
        log_file.write('\nParameters Names & Shapes: ')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                log_file.write('\n' + name + str(param.data.shape))

        log_file.write('\nTraining Dataset Size: ' + str(x_train.shape))
        log_file.write('\nValidation Dataset Size: ' + str(x_val.shape))

        log_file.write('\n' + get_class_dist(y_train, 'train'))
        log_file.write('\n' + get_class_dist(y_val, 'val') + '\n')

        log_file.close()
        
        train_file = open(self.save_dir + '/train_index.log', "w+")
        train_file.write(str(train_idx))
        train_file.close()
        val_file = open(self.save_dir + '/val_index.log', "w+")
        val_file.write(str(val_idx))
        val_file.close()

    def write_model_loss_metrics(self, epoch, loss, task):
        '''
        Write loss and task metrics to file (appending to meta-data info.log file)
        '''
        log_file = open(self.save_dir + '/info.log', "a+")
        log_file.write('\nEpoch: {:d} ------ {:s} ------'.format(epoch, task.upper()))
        log_file.write('\n{:s} loss: {:.4f}, '.format(task, loss) + str(self.metrics[task]))
        if task == 'val':
            log_file.write('\n')
        log_file.close()

    def write_best_metrics(self, metrics, train_loss, val_loss, epoch):
        log_file = open(self.save_dir + '/info.log', "a+")
        log_file.write('\nBEST METRICS FOR MODEL (acc. to {}) -------'.format(self.config['TRAINER']['monitor']))
        # Rounding the metrics for writing-
        for task in metrics:
            for key, value in metrics[task].items():
                metrics[task][key] = np.round(metrics[task][key], 4)
        log_file.write('\nEpoch: ' + str(epoch))
        log_file.write('\nMetrics: ' + str(metrics))
        log_file.write('\nTrain loss: {:.4f}'.format(train_loss))
        log_file.write('\nVal loss: {:.4f}'.format(val_loss))
        log_file.close()

    def logger(self, epoch, x_train, train_loss, val_loss): # no such file error here
        """
        Write to TensorBoard
        """
        #Writing to be done in the first epoch
        print('Epoch in TensorBoard:', epoch)
        self.writer['train'] = tb.SummaryWriter(log_dir=os.path.join(self.tb_path, 'train'))
        self.writer['val'] = tb.SummaryWriter(log_dir=os.path.join(self.tb_path, 'val'))
        if epoch==0:
            print('Writing to TensorBoard path', self.tb_path)
            if os.path.isdir(self.tb_path):
                shutil.rmtree(self.tb_path)

            sample_data = iter(self.trainloader).next()[0]  # [batch_size X seq_length X embedding_dim]
            print(sample_data.shape)
            self.writer['train'].add_graph(self.model, sample_data.to(self.device))
            self.writer['train'].add_text('Model:', str(self.model))
            self.writer['train'].add_text('Input shape:', str(x_train.shape))
            self.writer['train'].add_text('Data Preprocessing:', 'None, One-hot')
            self.writer['train'].add_text('Optimiser', str(self.optimizer))
            self.writer['train'].add_text('Batch Size:', str(self.config['DATA']['BATCH_SIZE']))
            self.writer['train'].add_text('Epochs:', str(self.config['TRAINER']['epochs']))

        for measure, value in self.metrics['train'].items():
            self.writer['train'].add_scalar(str('Train/'+measure), value, epoch)
        self.writer['train'].add_scalar('Loss', train_loss, epoch)
        for measure, value in self.metrics['val'].items():
            self.writer['val'].add_scalar(str('Val/'+measure), value, epoch)
        self.writer['val'].add_scalar('Loss', val_loss, epoch)

    def write_attention_maps(self, attention_maps, task, epoch):
        attention_map_path = self.att_path + '/' + task
        if not os.path.exists(attention_map_path):
            os.makedirs(attention_map_path)
        file_path = attention_map_path + '/attention_map_epoch{}'.format(epoch)

        atmaps = attention_maps.cpu().detach().numpy()
        atmaps = np.transpose(atmaps)
        #print('Writing Attention Map at {}...'.format(file_path))
        f=open(file_path,'a+')
        np.savetxt(f, atmaps, delimiter=",")
        f.close()


    def train_one_epoch(self, epoch, x_train, y_train):
        '''
        Train one epoch
        :param epoch: int - epoch number
        :param x_train: Numpy array - training data (sequences)
        :param y_train: Numpy array - training data (labels)
        :return: float - avg_train_loss
        '''

        # INPUT DATA
        print(len(x_train))
        trainset = SequenceDataset(x_train, y_train)  # NOTE: change input dataset size here if required
        self.trainloader = torch.utils.data.DataLoader(
                        trainset, batch_size=self.config['DATA']['BATCH_SIZE'],
                        shuffle=self.config['DATA']['SHUFFLE'], num_workers=self.config['DATA']['NUM_WORKERS'])

        # MODEL
        # self.model = SimpleLSTM(4,128,3,3)   [for eg.]
        self.model.to(self.device)
        print(self.device)

        # LOSS FUNCTION
        loss_fn = getattr(nn, self.config['LOSS'])()  # For eg: nn.CrossEntropyLoss()

        # OPTIMISER
        # self.optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)  # [for eg.] (or Adam)
        # in __init__
        

        # METRICS
        avg_train_loss = 0
        m = Metrics(self.config['TASK_TYPE'])   #m.metrics initialised to {0,0,0}
        self.metrics['train'] = m.metrics

        # FOR EACH BATCH
        for bnum, sample in tqdm(enumerate(self.trainloader)):

            self.model.train()
            self.model.zero_grad()
            #raw_out, attention_maps = self.model.forward(sample[0].to(f'cuda:{self.model.device_ids[0]}'))
            if self.config['MODEL_NAME']=="AttLSTM":
                raw_out, attention_maps = self.model.forward(sample[0].to(self.device))
            else:
                raw_out = self.model.forward(sample[0].to(self.device))
            labels = sample[1].long() #?
            if self.config['TASK_TYPE']=='regression':
                raw_out = raw_out.reshape(-1) #?
                labels = sample[1].float()
                
            loss = loss_fn(raw_out, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            # EVALUATION METRICS PER BATCH
            metrics_for_batch, pred = m.get_metrics(raw_out.detach().clone(), sample[1].detach().clone(), 'macro')  # todo: understand 'macro'
            #print('Predicted labels', pred)
            for key,value in metrics_for_batch.items():
                self.metrics['train'][key] += value
            avg_train_loss += loss.item()
            #self.write_attention_maps(attention_maps, 'train/all', epoch)
            
            if self.config['MODEL_NAME']=="AttLSTM":
                # Get Positive Label Attention Maps
                pos_label_indices = np.where(labels==1)[0].tolist()
                pos_attention_maps = attention_maps[:, pos_label_indices]
                self.write_attention_maps(pos_attention_maps, 'train/positive', epoch)

                # Get Negative Label Attention Maps
                neg_label_indices = np.where(labels == 0)[0].tolist()
                neg_attention_maps = attention_maps[:, neg_label_indices]
                self.write_attention_maps(neg_attention_maps, 'train/negative', epoch)

        # EVALUATION METRICS PER EPOCH
        for measure in m.metrics:
            self.metrics['train'][measure] /= (bnum+1)
        avg_train_loss /= (bnum+1)

        print('Epoch: {:d}, Train Loss: {:.4f}, '.format(epoch, avg_train_loss), self.metrics['train'])
        return avg_train_loss


    def val_one_epoch(self, epoch, x_val, y_val):
        '''
        Validation loop for epoch
        :param epoch: int - epoch number
        :param x_val: Numpy array - validation data (sequences)
        :param y_val: Numpy array - validation data (labels)
        :return: int - avg_val_loss
        '''

        valset = SequenceDataset(x_val, y_val)  # NOTE: change input dataset size here if required todo:
        val_dataloader = torch.utils.data.DataLoader(
                    valset, batch_size=self.config['DATA']['BATCH_SIZE'],
                    shuffle=self.config['DATA']['SHUFFLE'], num_workers=self.config['DATA']['NUM_WORKERS'])

        m = Metrics(self.config['TASK_TYPE'])  # m.metrics initialised to {0,0,0}
        self.metrics['val'] = m.metrics
        loss_fn =  getattr(nn, self.config['LOSS'])()
        avg_val_loss = 0    

        for bnum, sample in enumerate(val_dataloader):
            
            self.model.eval()
            if self.config['MODEL_NAME']=="AttLSTM":
                raw_out, attention_maps = self.model.forward(sample[0].to(self.device))
            else:
                raw_out = self.model.forward(sample[0].to(self.device))

            labels = sample[1].long()
            if self.config['TASK_TYPE'] == 'regression':
                raw_out = raw_out.reshape(-1)
                labels = sample[1].float() #need to change labels type for classification/regression
            loss = loss_fn(raw_out, labels.to(self.device))

            # EVALUATION METRICS PER BATCH
            metrics_for_batch, pred = m.get_metrics(raw_out.detach().clone(), sample[1].detach().clone(), 'macro')
            for key, value in metrics_for_batch.items():
                self.metrics['val'][key] += value
            avg_val_loss += loss.item()
            #self.write_attention_maps(attention_maps, 'val', epoch)


            if self.config['MODEL_NAME']=="AttLSTM":
            # Get Positive Label Attention Maps
                pos_label_indices = np.where(labels == 1)[0].tolist()
                pos_attention_maps = attention_maps[:, pos_label_indices]
                self.write_attention_maps(pos_attention_maps, 'val/positive', epoch)

                # Get Negative Label Attention Maps
                neg_label_indices = np.where(labels == 0)[0].tolist()
                neg_attention_maps = attention_maps[:, neg_label_indices]
                self.write_attention_maps(neg_attention_maps, 'val/negative', epoch)

        for measure in m.metrics:
            self.metrics['val'][measure] /= (bnum+1)
        avg_val_loss /= (bnum+1)

        print('Epoch: {:d}, Valid Loss: {:.4f}, '.format(epoch, avg_val_loss), self.metrics['val'])
        return avg_val_loss


    def training_pipeline(self):
        #Todo:For loading state, self.start epoch would change

        if not self.train_from_checkpoint:
            #self.encoded_seq = np.loadtxt(self.data_path + '/encoded_seq')
            no_timesteps = int(len(self.encoded_seq[0]) / 4)
            encoded_seq = self.encoded_seq.reshape(-1, no_timesteps, 4)
            print("Input data shape: ", encoded_seq.shape)
            y_label = np.loadtxt(self.data_path + '/' + self.y_label_filename)
            check_output_dim(self.config, y_label)
            
            if self.config['VALIDATION']['apply']:
                create_train_val_split = 'create_train_val_split_' + self.config['VALIDATION']['type']
                train_idx, val_idx = eval(create_train_val_split)(self.config['VALIDATION']['val_split'], y=y_label)

                # Create train/validation split ------
                self.x_train = encoded_seq[np.ix_(train_idx)] #replace `train_idx` by `np.arange(len(encoded_seq))` to use whole dataset
                self.y_train = y_label[np.ix_(train_idx)]
                self.x_val = encoded_seq[np.ix_(val_idx)]
                self.y_val = y_label[np.ix_(val_idx)]
                print(get_class_dist(self.y_train, 'train'))
                print(get_class_dist(self.y_val, 'val'))
            else:
                self.x_train = encoded_seq
                self.y_train = y_label
                print(get_class_dist(self.y_train, 'train'))

            self.write_model_meta_data(self.x_train, self.x_val, self.y_train, self.y_val, train_idx, val_idx)
        print(self.x_train.shape)

        # For early stopping calculation ---
        best_epoch = None
        monitor = 'val_loss' if self.config['TRAINER']['monitor'] == 'val_loss' \
            else "self.metrics['val']['"+self.config['TRAINER']['monitor']+"']"
        min_monitor = 99999999 if monitor == 'val_loss' else 0

        for epoch in range(self.start_epoch, self.config['TRAINER']['epochs']):
            print("Training Epoch %i -------------------" % epoch)

            epoch_tic = time.time()
            train_loss = self.train_one_epoch(epoch, self.x_train, self.y_train)

            if self.config['VALIDATION']['apply']:
                val_loss = self.val_one_epoch(epoch, self.x_val, self.y_val)
            if self.config['LR_SCHEDULER']['apply']:
                self.scheduler.step()

            epoch_toc = time.time()
            epoch_time = epoch_toc - epoch_tic

            print("******************* Epoch %i completed in %i seconds ********************" % (epoch, epoch_time))

            # Writing to be done in the first epoch
            if self.config['TRAINER']["save_model_to_dir"]:

                # SAVE TRAINING DETAILS TO INFO LOG
                self.write_model_loss_metrics(epoch, train_loss, 'train')
                if self.config['VALIDATION']['apply']:
                    self.write_model_loss_metrics(epoch, val_loss, 'val')

                # SAVE TO CHECKPOINT TO DIRECTORY
                if epoch % self.config['TRAINER']['save_period'] == 0:
                    self.save_checkpoint(epoch, save_type='any')

            # TENSORBOARD LOGGING
            if self.config['TRAINER']['tensorboard']:
                if not self.config['VALIDATION']['apply']:
                    val_loss = 0.0
                self.logger(epoch, self.x_train, train_loss, val_loss)
                # write to runs folder (create a model file name, and write the various training runs in it

            # EARLY STOPPING
            if eval(monitor) > min_monitor:
                # Save the model
                self.save_checkpoint(epoch=epoch, save_type='best')
                torch.save(self.model, self.save_dir + '/best_model'+str(epoch))
                epochs_no_improve = 0
                min_monitor = eval(monitor)  #todo be clear about what monitor keys can be used
                self.best_metrics = copy.deepcopy(self.metrics)
                self.best_metrics['train']['loss'] = train_loss
                self.best_metrics['val']['loss'] = val_loss
                self.best_metrics['train']['best_epoch'] = epoch
                print(self.best_metrics)
                best_epoch = epoch
            else:
                epochs_no_improve += 1
            if epoch > 10 and epochs_no_improve == self.config['TRAINER']['early_stop']:
                print('Early stopping!')
                print("Stopped after {:d} epochs".format(epoch))
                break

        # SAVE MODEL TO DIRECTORY
        if self.config['TRAINER']["save_model_to_dir"]:
            print('Saving model at ', self.save_dir)
            self.save_checkpoint(epoch=epoch, save_type='last')
            torch.save(self.model, self.save_dir+'/trained_model_'+self.model_name_save_dir+str(epoch))
            self.write_best_metrics(self.best_metrics, self.best_metrics['train']['loss'],
                                    self.best_metrics['val']['loss'], best_epoch)

        if self.config['TRAINER']['tensorboard']:
            self.writer['train'].close()
            self.writer['train'].close()

if __name__ == "__main__":

    chrm = "all/"
    
    config_file_name="config.json"
    projects_path="/ei/projects/a/abb983b7-84e2-4fbb-bce6-1af36c56f1a9/DDC/" # needed for running in scratch/

    if len(sys.argv)>1:
        config_file_name=sys.argv[1]
    
    print("Config file:", config_file_name)
    
    # Get config file
    with open(curr_dir_path + "/" + config_file_name, encoding='utf-8', errors='ignore') as json_data:
        config = json.load(json_data, strict=False)

    #Get System-specific Params file
    #with open('system_specific_params.yaml', 'r') as params_file:
    #    sys_params = yaml.load(params_file)

    #data_path = sys_params['DATA_WRITE_FOLDER']+'/'+chrm+config['DATASET_TYPE']+config["DATA"]["DATA_DIR"]
    data_path = projects_path + "data"+'/'+chrm+config['DATASET_TYPE']+config["DATA"]["DATA_DIR"]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('_%d-%m_%Y_%H.%M.%S')
    randomtag = "_"+str(randint(1,9999))
    model_name_save_dir = string_metadata(config) + timestamp + randomtag
    subdir = config['TRAINER']['sub_dir'] # e.g. "/all/att_start/"

    #model_dir = sys_params['LOGS_BASE_FOLDER'] + subdir + model_name_save_dir
    #tb_path = sys_params['RUNS_BASE_FOLDER'] + subdir + model_name_save_dir
    #att_path = sys_params['ATT_BASE_FOLDER'] + subdir + model_name_save_dir
    
    model_dir = "saved" + subdir + model_name_save_dir
    tb_path = "runs" + subdir + model_name_save_dir
    att_path = "attention" + subdir + model_name_save_dir

    config['TRAINER']['save_dir'] = model_dir
    config['TRAINER']['tb_path'] = tb_path
    config['TRAINER']['att_path'] = att_path

    obj = Training(config, model_name_save_dir, data_path, model_dir, tb_path, att_path) # has att_path
    obj.training_pipeline()


