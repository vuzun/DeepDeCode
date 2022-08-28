import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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

    def __init__(self, input_dims, out):
        super(SimpleLinear, self).__init__()
        self.input_dims=input_dims
        self.linear=nn.Linear(input_dims, out)
        self.relu=nn.ReLU()
        
    
    def forward(self, x):
        x=x.reshape(-1,self.input_dims)
        x=self.linear(x)
        x=self.relu(x)
        return x

''' ***** Simple LSTM ***** '''
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
                                                    # pass batch_size as a parameter incase of incomplete batch
        lstm_out, (h_n, c_n) = self.lstm(input, hidden)
        #concat_state = torch.cat((lstm_out[:, -1, :self.hidden_units], lstm_out[:, 0, self.hidden_units:]), 1)
        # same as `output = output[:, -1, :]`?
        hidden_reshape = h_n.reshape(-1, self.hidden_units * self.num_directions * self.hidden_layers) # hidden_layers?, doesn't this break with N>1?

        # why take all hidden states?

        raw_out = self.output_layer(hidden_reshape)
        #raw_out = self.output_layer(h_n[-1])

        return raw_out

''' ***** CNN ***** '''
class CNN(BaseModel):
    def __init__(self, no_classes, device):
        super(CNN, self).__init__()
        self.no_classes = no_classes
        self.device = device

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=(5,15), stride=1, padding=2),
            #nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=(5,15), stride=1, padding=2),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout
            #nn.Dropout()
        )
        # Defining fully-connected layer
        #self.linear_layers = nn.Sequential(nn.Linear(self.cnn_layers.shape[1], self.no_classes))

    def linear_layer(self, outputlength):
        linear_layer = nn.Sequential(nn.Linear(outputlength, 1000).to(self.device))
        return linear_layer

    def forward(self, x):
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)

        outputlength = x.size()[1]
        linear_layers = self.linear_layer(outputlength)
        output = linear_layers(x)
        output1 = nn.Linear(1000, self.no_classes).to(self.device)(output)
        return output1


''' ***** Attention Model ***** '''

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


class rec_attention(nn.Module):
    # attention with bin context vector
    def __init__(self, args):
        super(rec_attention, self).__init__()
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
    # modular LSTM encoder
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
        self.rec_attention = rec_attention(args)

    def total_h_size(self):
        return self.total_hidden_size

    def forward(self, seq, hidden=None):
        torch.manual_seed(0) # ?

        lstm_out, (h_n, c_n) = self.lstm(seq, hidden)
        h_n = h_n.reshape(-1, self.hidden_size * self.num_directions * self.num_layers)

        bin_output_for_att = lstm_out.permute(1, 0, 2) # N, L, DHout > L, N, DHout
        nt_rep, bin_alpha = self.rec_attention(bin_output_for_att)
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

        #[batch_size, _, _] = iput.size() does nothing?
        att_rep, hidden_reshape, bin_a = self.encoder(iput) # hidden_reshape will change based on stacked num???
        #print(f"In att_DNA, rep before squeeze: {att_rep.shape}")
        att_rep = att_rep.squeeze(1)
        #print(f"In att_DNA, lvl1rep: {att_rep.shape}, hidden_reshape: {hidden_reshape.shape}")
        concat = torch.cat((hidden_reshape, att_rep), dim=1)
        bin_pred = self.linear(concat) # 3 layers error, 2 no
        sigmoid_pred = torch.sigmoid(bin_pred)
        return sigmoid_pred, bin_a

