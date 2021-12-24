# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:33:47 2021

@author: Fokhrul
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 8, stride= 1, padding=1)
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride= 1)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6)
        
        self.optim = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation):
        observation = torch.tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128*19*8)
        observation = F.relu(self.fc1(observation))
        
        actions = self.fc2(observation)
        
        return actions
    
class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 MaxMemSize, epsEnd = 0.05,
                 replace = 10000, actionSpace = [i for i in range(6)]):
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsEnd = epsEnd
        self.actionSpace = actionSpace
        self.memSize = MaxMemSize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_terget_cnt = replace
        self.Q_eval = DeepQNetwork(self.alpha)
        self.Q_next = DeepQNetwork(self.alpha)
    
    
    
    
    