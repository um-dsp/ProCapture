# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:48:36 2023

@author: aamich
"""

import torch.nn as nn
import torch.nn.functional as F
import ember
import keras
import tensorflow as tf
import numpy as np
import os


class NeuralNetMnist_1(nn.Module):
    '''Feature extraction model for mnist1'''
    
    def __init__(self):

        super(NeuralNetMnist_1, self).__init__()
        self.conv1 = nn.Conv1d( 2,64,kernel_size =3)
        self.conv2 = nn.Conv1d( 64,64,kernel_size =3)
        self.conv3= nn.Conv1d(64,1,kernel_size =3)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(599 , 2)
        self.fc2 = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x) 
        x=self.conv3(x)
        x=F.relu(x) 
        x= self.mp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        #output = F.log_softmax(x, dim=1)
        
        return output

class NeuralNetMnist_2(nn.Module):
    '''Feature extraction model for mnist2'''
    def __init__(self):

        super(NeuralNetMnist_2, self).__init__()
        self.conv1 = nn.Conv1d( 1,64,kernel_size =4)
        self.conv2= nn.Conv1d(64,32,kernel_size =4)
        self.conv3= nn.Conv1d(32,2,kernel_size =3)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(193 , 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x) 
        x=self.conv3(x)
        x=F.relu(x)
        x= self.mp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output

        
class NeuralNetCuckoo_1(nn.Module):
    '''Feature extraction model for Cuckoo_1'''
    def __init__(self):

        super(NeuralNetCuckoo_1, self).__init__()
        self.conv1 = nn.Conv1d( 1,64,kernel_size =4)
        self.conv2 = nn.Conv1d( 64,64,kernel_size =4)
        self.conv3= nn.Conv1d(64,1,kernel_size =4)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(27 , 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x) 
        x=self.conv3(x)
        x=F.relu(x) 
        x= self.mp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output

class NeuralNetMnist_3(nn.Module):
    '''Feature extraction model for mnist3'''
    def __init__(self):

        super(NeuralNetMnist_3, self).__init__()
        self.conv1 = nn.Conv1d( 1,64,kernel_size =4)
        self.conv2 = nn.Conv1d( 64,64,kernel_size =4)
        self.conv3= nn.Conv1d(64,1,kernel_size =4)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 , 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x) 
        x=self.conv3(x)
        x=F.relu(x) 
        x= self.mp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output



class NeuralNetCifar_10(nn.Module):
    '''Feature extraction model for cifar10'''
    def __init__(self):

        super(NeuralNetCifar_10, self).__init__()
        self.conv1 = nn.Conv1d( 1,64,kernel_size =4)
        self.conv2 = nn.Conv1d( 64,64,kernel_size =4)
        self.conv3= nn.Conv1d(64,1,kernel_size =4)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.drop = nn.Dropout(0.4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1952 , 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.relu(x) 
        x=self.conv3(x)
        x=F.relu(x) 
        x= self.mp(x)
        x = self.drop(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output