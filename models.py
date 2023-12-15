import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import math
import time
from torch.nn.functional import one_hot


# simple multiple layer fully connected network with ReLU activation and dropout
class MLP(nn.Module):
    def __init__(self, d_input, d_hide, d_output=30 ,n_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(d_input, d_hide)
        self.layers = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(d_hide), nn.Linear(d_hide, d_hide)) for _ in range(n_layers)])        
        self.last_layer = nn.Linear(d_hide, d_output)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1) # flatten
        x = self.first_layer(x)
        for layer in self.layers:
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = layer(x)
        x = nn.functional.relu(x)
        x = self.last_layer(x)
        return x

class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""
    def __init__(self,
                 embed_dim=300,
                 filter_sizes=[5],
                 num_filters=[100],
                 num_classes=2,
                 num_layers=3,
                 dropout=0.1):
        """
        The constructor for CNN_NLP class.

        Args:
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        self.embed_dim = embed_dim
        num_filters *= num_layers
        filter_sizes *= num_layers

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i],
                      stride = 5)
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = input_ids.permute(0, 2, 1).float()

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits
    
class MyLinear(nn.Module):

    def __init__(
        self, input_dim, out_dim, bias=False
    ):
        """
        Args:
            input_dim: The input dimension.
            out_dim: The output dimension.
            bias: True for adding bias.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn( out_dim, input_dim)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            An affine transformation of x, tensor of size (batch_size, *, out_dim)
        """
        x = F.linear( x, self.weight, self.bias) / x.size(-1)**.5 # standard scaling
        return x

class MYMLP(nn.Module):
    def __init__(
        self, d_input, d_hide, d_output, n_layers, bias=True, norm='mf',dropout=0.1
    ):
        """
        MultiLayer Perceptron

        Args:
            input_dim: The input dimension.
            nn_dim: The number of hidden neurons per layer.
            out_dim: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()
        input_dim = d_input
        nn_dim =d_hide
        out_dim = d_output
        num_layers = n_layers
        self.hidden = nn.Sequential(
            nn.Sequential(
                MyLinear(
                    input_dim, nn_dim, bias
                ),
                nn.ReLU(),
            ),
            *[nn.Sequential(
                    MyLinear(
                        nn_dim, nn_dim, bias
                    ),
                    nn.ReLU(),
                )
                for _ in range(1, num_layers)
            ],
        )
        self.readout = nn.Parameter(
            torch.randn(nn_dim, out_dim)
        )
        if norm=='std':
            self.norm = nn_dim**.5 # standard NTK scaling
        elif norm=='mf':
            self.norm = nn_dim # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            Output of a multilayer perceptron, tensor of size (batch_size, *, out_dim)
        """
        x = x.view(x.size(0), -1) # flatten
        x = self.hidden(x)
        x = x @ self.readout / self.norm
        return x