
import re
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging
import botorch
from dataloader_base import PatientDataset
from transformers import AutoTokenizer, AutoModel


class mmdp(nn.Module):
    def __init__(self):
        super(mmdp, self).__init__()
        self.hidden_size = 768
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.t_tok = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.l_toq = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.t_value = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)


        self.drop_out1 = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.3)
        self.drop_out3 = nn.Dropout(0.3)
        self.drop_out4 = nn.Dropout(0.3)
        self.drop_out5 = nn.Dropout(0.3)

        self.ff = nn.Sequential(
                    nn.Linear(self.hidden_size//2, self.hidden_size//2),
                    nn.PReLU(),
                    nn.Linear(self.hidden_size//2, self.hidden_size//2)
                    )
        
        self.MLPs = nn.Sequential(
                    nn.Linear(self.hidden_size//2, 100),
                    nn.Dropout(0.3),
                    nn.Linear(100, 3),
                    )
        self.phrase_filter = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(3,1))
        self.phrase_extract = nn.MaxPool2d(kernel_size=(1, 25))

        self.sigmoid = nn.Sigmoid()


    
    def approximation(self, Ot):
        text_embedding = self.text_encoder(**Ot).last_hidden_state

        value_t = self.drop_out3(self.t_value(text_embedding))

        return value_t.mean(1)


    def forward(self, fuse_input):

        u = self.approximation(fuse_input)
        return u
        # y =  self.sigmoid(self.MLPs(u))
        # return y



       

    


   