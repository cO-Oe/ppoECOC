import math
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.num_classes = envs.single_action_space.shape[0]
        obs_shape = envs.single_observation_space.shape[0]
        self.confusion_matrix_size = self.num_classes * self.num_classes
        self.ecoc_matrix_size = obs_shape - self.confusion_matrix_size
        
        # separate embedding layers for confusion matrix and ECOC matrix
        self.confusion_embedding = nn.Linear(self.confusion_matrix_size, 64)
        self.ecoc_embedding = nn.Linear(self.ecoc_matrix_size, 64)
        
        # combine embeddings
        self.combine_embedding = nn.Linear(128, 128)
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(128, dropout=0.1, max_len=500)
        
        # transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=3)
        
        # output layers (actor & critic)
        self.actor = layer_init(nn.Linear(128, self.num_classes * 3), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, x):
        # split input into confusion matrix and ECOC matrix
        confusion_matrix = x[:, :self.confusion_matrix_size]
        ecoc_matrix = x[:, self.confusion_matrix_size:]
        
        # process confusion matrix and ECOC matrix separately
        confusion_embed = self.confusion_embedding(confusion_matrix)
        ecoc_embed = self.ecoc_embedding(ecoc_matrix)
        
        # combine embeddings
        combined = torch.cat((confusion_embed, ecoc_embed), dim=1)
        x = self.combine_embedding(combined).unsqueeze(1)  # Add sequence dimension
        
        # apply positional encoding and transformer
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x.squeeze(1)  # Remove sequence dimension

    def get_value(self, x):
        x = self.forward(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.forward(x)

        # generate action logits and reshape for multi-class actions
        action_logits = self.actor(x).view(-1, self.num_classes, 3)

        # compute action probabilities using softmax
        action_probs = torch.softmax(action_logits, dim=-1)

        # create a categorical distribution based on action probabilities
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        # return action, log probability, entropy, and value from the critic
        action_values = action 
        return action_values, dist.log_prob(action).sum(dim=-1), dist.entropy().sum(dim=-1), self.critic(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        # create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # calculate scaling factors for dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape and register as a buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)