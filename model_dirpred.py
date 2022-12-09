
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
from transformers import AutoTokenizer, AutoModel
from numpy.testing import assert_almost_equal

class mmdp(nn.Module):
    def __init__(self,class_3,latent_ndims,n_tokens):
        super(mmdp, self).__init__()
        self.class_3 = class_3
        self.hidden_size = 768
        self.latent_ndims = latent_ndims
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.t_tok = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.l_toq = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)
        self.t_value = nn.Linear(self.hidden_size,self.hidden_size//2,bias=False)


        self.init_lab_prompt_value = self.text_encoder.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
        self.soft_lab_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_lab_prompt.weight = nn.parameter.Parameter(self.init_lab_prompt_value)
        
        self.init_event_prompt_value = self.text_encoder.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
        self.soft_event_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_event_prompt.weight = nn.parameter.Parameter(self.init_event_prompt_value)

        self.init_text_prompt_value = self.text_encoder.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
        self.soft_text_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_text_prompt.weight = nn.parameter.Parameter(self.init_text_prompt_value)

        self.init_label_prompt_value = self.text_encoder.embeddings.word_embeddings.weight[:n_tokens].clone().detach()
        self.soft_label_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_label_prompt.weight = nn.parameter.Parameter(self.init_label_prompt_value)

        self.drop_out1 = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.3)
        self.drop_out3 = nn.Dropout(0.3)
        self.drop_out4 = nn.Dropout(0.3)
        self.drop_out5 = nn.Dropout(0.3)
        self.drop_out6 = nn.Dropout(0.3)
        self.drop_out7 = nn.Dropout(0.3)
    
      
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


        self.sigmoid = nn.Sigmoid()
        self.aplpha_fc = nn.Linear( self.hidden_size//2,  latent_ndims)
        self.beta_fc = nn.Linear( self.hidden_size//2,  latent_ndims)  
        self.cluster_fc = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.PReLU()
            # nn.Tanh()
            )


        self.transd = nn.GRU(input_size = self.hidden_size//2, batch_first=True, hidden_size =  self.hidden_size//2, dropout= 0.3,num_layers=1, bidirectional=True)

        self.prior_beta = nn.Sequential(
            nn.Linear(self.hidden_size//2, latent_ndims),
            nn.PReLU(),
            nn.Linear(latent_ndims, 1),
            )

        self.transRNN =  nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size = self.hidden_size//2, dropout= 0.3,num_layers=1, bidirectional=True)

     
        self.forget_gate =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            )
        self.Ztd_cat = nn.Linear(self.hidden_size, self.hidden_size//2)



    def cross_attention(self,v,c):
       
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        g = torch.bmm(v, c.transpose(-2, -1))
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]
        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        return b

    
    def reparametrize(self, param1, param2,uniform_distribution, parametrization=None):

        # for Kumaraswamy, param1 == alpha, param2 == beta
        v0 = self.get_kumaraswamy_samples(param1, param2,uniform_distribution)
        v = self.set_v_K_to_one(v0)
        # v = self.get_kumaraswamy_samples(param1, param2,uniform_distribution)
        out = self.get_stick_segments(v)

        return out,v
    
    def get_kumaraswamy_samples(self, param1, param2,uniform_distribution):
        # u is analogous to epsilon noise term in the Gaussian VAE
 

        u = uniform_distribution.sample().squeeze()
        v = (1 - u.pow(1 / param2)).pow(1 / param1)
        return v  # sampled fractions


    def set_v_K_to_one(self, v):
        # set Kth fraction v_i,K to one to ensure the stick segments sum to one
        if v.ndim > 2:
            v = v.squeeze()
        v0 = v[:, -1].pow(0).reshape(v.shape[0], 1)
        v1 = torch.cat([v[:, :self.latent_ndims - 1], v0], dim=1)
        # v = F.softmax(v)
        return v1


    def get_stick_segments(self, v):
        n_samples = v.size()[0]
        n_dims = v.size()[1]


        pi = torch.ones((n_samples, n_dims))
        for k in range(n_dims):
            if k == 0:
                pi[:, k] = v[:, k]
            else:
                pi[:, k] = v[:, k] * torch.stack([(1 - v[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)
        # print(v[:1,:])
        # print(pi[:1,:])
        # print(pi[:1,:].sum())

        # ensure stick segments sum to 1
        assert_almost_equal(torch.ones(n_samples), pi.sum(axis=1).detach().numpy(),
                            decimal=2, err_msg='stick segments do not sum to 1')


        return pi


    def approximation(self,Ztd_list,fuse_input):

       
        text_embedding = self.text_encoder(**fuse_input).last_hidden_state
        value_t = self.drop_out3(self.t_value(text_embedding))
        value_t = value_t.mean(1)

        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        Ztd_last =  torch.mean(Ztd_last,0)
        Ztd = torch.cat((Ztd_last,value_t),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)
        Ztd = self.drop_out6( self.Ztd_cat(gate_ratio_ztd*Ztd))

        return Ztd

   
    def trasition(self,Ztd_last):
    

        _,Ztd_last_last_hidden = self.transd(Ztd_last.unsqueeze(0))

        Ztd =  torch.mean(Ztd_last_last_hidden,0)

        Beta =   F.softplus(self.prior_beta(Ztd))

        return Beta
      

    def forward(self,Ztd_list,Ztd_last, fuse_input,center_embedding,uniform_distribution):
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)

        z = self.approximation(Ztd_list, fuse_input)
        prior_beta_ = self.trasition(Ztd_last)

        alpha = F.softplus(self.aplpha_fc(z))
        beta  = F.softplus(self.beta_fc(z))
        pi,v = self.reparametrize(alpha, beta, uniform_distribution)
        ck = self.drop_out7(self.cluster_fc(center_embedding))
        u = torch.matmul(pi.to(center_embedding.device), ck)
        y =  self.sigmoid(self.MLPs(u))
        return y,alpha,beta,prior_beta_,u,v,pi




       

    


   