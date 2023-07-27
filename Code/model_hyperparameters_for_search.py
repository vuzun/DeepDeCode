# collection of hyperparameters for different models

# Linear
h_hidden_dims_linear=[10,20,50,100,1000]
linear_hyperparam_list=[h_hidden_dims_linear]


# LSTM
h_batch_size = [32, 64, 128, 256]
h_hidden_dims = [8, 16, 32, 64, 128]
h_num_layers = [1, 2, 3]
h_num_fc_layers=[] # 1,2
h_num_fc_neurons=[] # 12, 32, 64, #output
h_lr = [0.1, 0.01, 0.001, 0.0001]
h_dropout = [0.0, 0.2, 0.5]

lstm_hyperparam_list = [h_batch_size, h_hidden_dims, h_num_layers, h_lr, h_dropout]


# CNN
h_batch_size=[] # 32, 64, 128, 256
h_num_conv_layers=[] # 1, 2, 3
h_num_fc_layers=[] # 1, 2

h_filter_num=[8, 16, 32, 64, 128]
h_kernel_size=[5, 6, 7] # 5x5, 6x6, 7x7, but PyTorch makes it symmetrical by default
h_conv_layer_stride=[1, 2]
h_max_pool_kernel_size=[2, 3]
h_max_pool_stride=[1, 2]
h_fc_neuron_num=[16, 32, 64, 1000]

cnn_hyperparam_list = [h_filter_num, h_kernel_size, h_conv_layer_stride, h_max_pool_kernel_size, h_max_pool_stride, h_fc_neuron_num]

# variable names for model types
# config["MODEL"] to be updated based on these
model_2_var_dict={
    "AttLSTM": ['BATCH_SIZE', 'hidden_dim', 'hidden_layers', 'lr', 'dropout'],
    "SimpleLSTM": ['BATCH_SIZE', 'hidden_dim', 'hidden_layers', 'lr', 'dropout'],
    "CNN": ["filter_num", "kernel_size", "conv_layer_stride",
        "max_pool_kernel_size", "max_pool_stride",
        "fc_neuron_num"],
    "SimpleLinear": ['hidden_dim_linear']
}

model_2_hp_list_dict={
    "AtttLSTM": lstm_hyperparam_list,
    "SimpleLSTM": lstm_hyperparam_list,
    "CNN": cnn_hyperparam_list,
    "SimpleLinear": linear_hyperparam_list
}
