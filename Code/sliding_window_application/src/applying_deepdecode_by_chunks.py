import pandas as pd
import numpy as np
from statistics import mean
import torch
from torch._C import TensorType
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys

from models import *

# assuming a command line argument passed

try:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    print(arg1)
except IndexError:
    print("Missing argument!")
    sys.exit(1)

# arg1="encoded_10khead_1step.txt"
# arg2="rez/outputchr12.pt"
# arg3="best_cpoint_butC12.pth"

#y_label_path="C:/Users/uzunv/Desktop/asmitaSeptember/Model14_AttLSTM[4,16,2,2]_BS4_LR0.001_27-08_2021_11_40/y_label_val"
chr_encoded_seq=arg1 #"raw/by100k_step25_chr"+arg2+"/encoded_till"+arg1+"00k_by100k_s25"
output_path= arg2#"results/output_chr"+arg2+"_till"+arg1+"00k_by100k_25s.pt"
checkpoint_path=arg3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Torch using",device)


# a dict, eopch, state_dict, optimizer, config
mcheck=torch.load(checkpoint_path, map_location=device)
#print(mcheck)

mcheck['config']['MODEL_NAME'] # AttLSTM
mcheck['config']["OPTIMIZER"]  # Adam

config=mcheck['config']

#encoded_seq=np.loadtxt("encoded_seq_val") # 13551 x 400?
encoded_seq=np.loadtxt(chr_encoded_seq)
#y_label = np.loadtxt(y_label_path)
no_timesteps = int(len(encoded_seq[0]) / 4) # ???
encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4) #???

# tailored to AttLSTM, not others!
if config['MODEL_NAME']=="AttLSTM":
    args = {'n_nts': config['MODEL']['embedding_dim'], 'n_bins': encoded_seq.shape[1],  # 100
    'bin_rnn_size': config['MODEL']['hidden_dim'], 'num_layers': config['MODEL']['hidden_layers'],
    'dropout': config['TRAINER']['dropout'], 'bidirectional': config['MODEL']['bidirectional']}
    model = att_DNA(args, config['MODEL']['output_dim'])

if config['MODEL_NAME']=='SimpleLSTM':
    model = eval(config['MODEL_NAME'])(config['MODEL']['embedding_dim'], config['MODEL']['hidden_dim'],
                                        config['MODEL']['hidden_layers'], config['MODEL']['output_dim'],
                                        config['DATA']['BATCH_SIZE'], config['MODEL']['bidirectional'],
                                        config['TRAINER']['dropout'], device)
if config['MODEL_NAME']=='CNN':
    model = eval(config['MODEL_NAME'])(config['MODEL']['output_dim'], device)

model.to(device)

model.load_state_dict(mcheck['state_dict'])
optimizer=getattr(optim, config['OPTIMIZER']['type']) \
    (model.parameters(), lr=config['OPTIMIZER']['lr'], weight_decay=config['OPTIMIZER']['weight_decay'])
optimizer.load_state_dict(mcheck["optimizer"])

# from train_utils.py
class SequenceDataset(Dataset):

    def __init__(self, data, labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index
        return (self.data[idx], self.labels[idx])


testset = SequenceDataset(encoded_seq, np.array([0 for _ in range(len(encoded_seq))]))#y_label)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=4)#len(testset))

# loss_fn = nn.CrossEntropyLoss()

# batch it
result_class=torch.tensor([]) #, device=device) # CUDA out of memory - up to 10GB it seems
result_att=torch.tensor([]) #, device=device)
model.eval()


for bnum, datatuple in enumerate(test_dataloader):
    if bnum%250==0:
        print(bnum*4)
    data, labels = datatuple #iter(test_dataloader).next() 
    # for whole chr19, need 60GB ram? > chunks
    output_of_model=model.forward(data.to(device))
    if config['MODEL_NAME']=="AttLSTM":
        result_class=torch.cat((result_class, output_of_model[0].to('cpu')), 0)
        result_att=torch.cat((result_att, output_of_model[1].to('cpu')), 1)
    else:
        result_class=torch.cat((result_class, output_of_model.to('cpu')), 0)

dir_name=os.path.dirname(output_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

if config['MODEL_NAME']=="AttLSTM":
    torch.save((result_class, result_att), output_path)
else:
    torch.save(result_class, output_path)
