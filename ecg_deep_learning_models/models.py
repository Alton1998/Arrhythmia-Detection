import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator


class ECGANNModel(nn.Module):
    def __init__(self, input_size=200, layers=[100], out_size=5, p=0.5):
        super().__init__()
        layer_list = []
        n_in = input_size
        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Dropout(p))
            n_in = i

        layer_list.append(nn.Linear(layers[-1], out_size))

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layers(x)
        return x

class ECGCNNModel(nn.Module):
    def __init__(self,fc_layers=[100,50],kernel_size=3,stride=1,max_pool_size=2,input_dim=200,out_sz=5,p=0.5):
        super().__init__()
        cnn_layer_list = []
        fc_layer_list = []
        cnn_layer_list.append(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=kernel_size,stride=stride))
        cnn_layer_list.append(nn.ReLU(inplace=True))
        cnn_layer_list.append(nn.Conv1d(in_channels=16,out_channels=32,kernel_size=kernel_size,stride=stride))
        cnn_layer_list.append(nn.ReLU(inplace=True))
        cnn_layer_list.append(nn.MaxPool1d(max_pool_size,stride=stride))
        cnn_layer_list.append(nn.BatchNorm1d(32))
        cnn_layer_list.append(nn.Conv1d(in_channels=32,out_channels=64,kernel_size=kernel_size,stride=stride))
        cnn_layer_list.append(nn.ReLU(inplace=True))
        cnn_layer_list.append(nn.Conv1d(in_channels=64,out_channels=128,kernel_size=kernel_size,stride=stride))
        cnn_layer_list.append(nn.ReLU(inplace=True))
        cnn_layer_list.append(nn.MaxPool1d(max_pool_size,stride=stride))
        cnn_layer_list.append(nn.BatchNorm1d(128))
        cnn_layer_list.append(nn.Flatten())
        self.conv_layer = nn.Sequential(*cnn_layer_list)
        n_in = 128 *190
        for i in fc_layers:
            fc_layer_list.append(nn.Linear(n_in, i))
            fc_layer_list.append(nn.ReLU(inplace=True))
            fc_layer_list.append(nn.Dropout(p))
            n_in = i

        fc_layer_list.append(nn.Linear(fc_layers[-1], out_sz))

        self.fc_layers = nn.Sequential(*fc_layer_list)

    def forward(self,x):
        x = self.conv_layer(x)
        x = self.fc_layers(x)
        return x