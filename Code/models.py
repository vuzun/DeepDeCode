import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class SimpleLinear(BaseModel):

    def __init__(self, input_dims, hidden_dims, output_dim):
        super(SimpleLinear, self).__init__()
        self.input_dims=input_dims   
        self.linlayers=nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.linear(hidden_dims, output_dim),
            nn.Softmax()
        )

    def forward(self, x):
        x=x.reshape(-1,self.input_dims)
        x=self.linlayers(x)
        return x


class SimpleLSTM(BaseModel):

    def __init__(self, input_dims, hidden_units, hidden_layers, out, batch_size,
                 bidirectional, dropout, device):
        super(SimpleLSTM, self).__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.device = device
        self.dropout=dropout
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=self.input_dims, hidden_size=self.hidden_units, num_layers=self.hidden_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        self.output_layer = nn.Linear(self.hidden_units * self.num_directions * self.hidden_layers, out)

    def init_hidden(self, batch_size):
        torch.manual_seed(0)
        hidden = torch.rand(self.num_directions*self.hidden_layers, batch_size, self.hidden_units,
                            device=self.device, dtype=torch.float32)
        cell = torch.rand(self.num_directions* self.hidden_layers, batch_size, self.hidden_units,
                          device=self.device, dtype=torch.float32)

        hidden = nn.init.xavier_normal_(hidden)
        cell = nn.init.xavier_normal_(cell)

        return (hidden, cell)

    def forward(self, input):
        hidden = self.init_hidden(input.shape[0])  # tuple containing hidden state and cell state; `hidden` = (h_t, c_t)
        lstm_out, (h_n, c_n) = self.lstm(input, hidden)
        hidden_reshape = h_n.reshape(-1, self.hidden_units * self.num_directions * self.hidden_layers)
        raw_out = self.output_layer(hidden_reshape)

        return raw_out



class CNN(BaseModel):
    def __init__(self, no_classes, device, batch_size=1,
            conv_layer_num=2, filter_num=32, kernel_size=5, conv_layer_stride=1,
            max_pool_kernel_size=2, max_pool_stride=2,
            fc_layer_num=None, fc_neuron_num=1000,
            dropout=0.0):
        super(CNN, self).__init__()
        self.no_classes = no_classes
        self.device = device
        self.batch_size = batch_size
        
        self.conv_layer_num=conv_layer_num
        self.filter_num=filter_num
        self.kernel_size=kernel_size
        self.conv_layer_stride=conv_layer_stride

        self.max_pool_kernel_size=max_pool_kernel_size
        self.max_pool_stride=max_pool_stride
        
        self.fc_layer_num=fc_layer_num
        self.fc_neuron_num=fc_neuron_num
        self.dropout=dropout

  
        self.cnn_layers = nn.Sequential(
        
            nn.Conv2d(1, self.filter_num, self.kernel_size, self.conv_layer_stride, padding=2),
            #nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_kernel_size, self.max_pool_stride),  #! max_pooling_kernel_size, stride
            nn.Conv2d(self.filter_num, 64, self.kernel_size, self.conv_layer_stride, padding=2),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_kernel_size, self.max_pool_stride),
            nn.Dropout(self.dropout)
        )
        
        
        cnn_layers_output=self.cnn_layers(torch.rand(1,1,4,100)) # 4 - encoding, 100 - subsequence length
        cnn_layers_outputlength=cnn_layers_output.view(1,-1).size(1) # view to collapse dimensions after samples/N
        
        self.linear_layers = nn.Sequential(
            nn.Linear(cnn_layers_outputlength, self.fc_neuron_num).to(self.device),
            nn.Linear(self.fc_neuron_num, self.no_classes).to(self.device)
        )

    def forward(self, x):
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        x = x.unsqueeze(1)
        
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)

        output = self.linear_layers(x)
        
        return output


# Attention and combination classes

def batch_product(iput, cvec):
    result = None
    for i in range(iput.size()[0]):
        op = torch.mm(iput[i], cvec)
        op = op.unsqueeze(0)
        if (result is None):
            result = op
        else:
            result = torch.cat((result, op), 0)
    return result.squeeze(2)


class attention_layer(nn.Module):
    # attention layer
    def __init__(self, args):
        super(attention_layer, self).__init__()
        self.num_directions = 2 if args['bidirectional'] else 1
        self.bin_rep_size = args['hidden_size'] * self.num_directions

        self.bin_context_vector = nn.Parameter(torch.Tensor(self.bin_rep_size, 1), requires_grad=True) # (D*h_size) * 1
        self.bin_context_vector.data.uniform_(-0.1, 0.1)  # Learnable parameter
        self.softmax = nn.Softmax(dim=1)


    def forward(self, iput):
        alpha = self.softmax(batch_product(iput, self.bin_context_vector)) # L * batch << (L,N,DHout x DHsize,1)
        [seq_length, batch_size, total_hidden_size] = iput.size()
        repres = torch.bmm(alpha.unsqueeze(2).view(batch_size, -1, seq_length),
                           iput.view(batch_size,seq_length,total_hidden_size))
                           
        return repres, alpha


class recurrent_encoder(nn.Module):
    # LSTM + attention layer
    def __init__(self, seq_length, input_bin_size, args):
        super(recurrent_encoder, self).__init__()
        self.hidden_size = args['hidden_size']
        self.num_layers = args['num_layers']
        self.input_dims = input_bin_size
        self.seq_length = seq_length

        self.num_directions = 2 if args['bidirectional'] else 1
        self.total_hidden_size = self.hidden_size * self.num_directions
        
        self.lstm = nn.LSTM(self.input_dims, self.hidden_size, num_layers=self.num_layers, dropout=args['dropout'],
                           bidirectional=args['bidirectional'], batch_first=True)
        self.attention_layer = attention_layer(args)

    def total_h_size(self):
        return self.total_hidden_size

    def forward(self, seq, hidden=None):
        torch.manual_seed(0)

        lstm_out, (h_n, c_n) = self.lstm(seq, hidden)
        h_n = h_n.reshape(-1, self.hidden_size * self.num_directions * self.num_layers)

        bin_output_for_att = lstm_out.permute(1, 0, 2) # N, L, DHout > L, N, DHout
        nt_rep, bin_alpha = self.attention_layer(bin_output_for_att)
        #print(f"lstm_out: {lstm_out.shape}")
        #print(f"nt_rep size: {nt_rep.shape}, bin_alpha size: {bin_alpha.shape}, h_c : {h_n.shape}")
        return nt_rep, h_n, bin_alpha

class att_DNA(BaseModel):
    def __init__(self, args, out):
        super(att_DNA, self).__init__()
        self.n_nts = args['n_nts']
        self.seq_length = args['n_bins']
        self.encoder = recurrent_encoder(self.seq_length, self.n_nts, args)
        self.outputsize = self.encoder.total_h_size()
        self.linear = nn.Linear(self.outputsize * (1+args['num_layers']), out) # 3 assumes 2 hidden layers and 1 for repres

    def forward(self, iput):
        att_rep, hidden_reshape, bin_a = self.encoder(iput)
        #print(f"In att_DNA, rep before squeeze: {att_rep.shape}")
        att_rep = att_rep.squeeze(1)
        #print(f"In att_DNA, lvl1rep: {att_rep.shape}, hidden_reshape: {hidden_reshape.shape}")
        concat = torch.cat((hidden_reshape, att_rep), dim=1)
        bin_pred = self.linear(concat) 
        sigmoid_pred = torch.sigmoid(bin_pred)
        return sigmoid_pred, bin_a

