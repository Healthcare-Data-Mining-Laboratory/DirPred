import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_base import PatientDataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from model_center_embedding import mmdp
from sklearn.cluster import KMeans
import joblib
from transformers import AutoTokenizer
import copy

SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="3"


## scigpu10  text_event_lab

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
SEED = 2019

visit = 'once'
class_3 = True
max_length = 300
BATCH_SIZE = 30
number_cluster = 16
weight_dir = "xx.pth"
device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)

start_epoch = 0




def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]
    seq_mask = attention_mask[:,[-1]]
    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - input_ids.shape[1]
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)
    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)

    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec

def cat_feature(max_length,text,event_labs):
    text_input_ids = text['input_ids'][:,:max_length-1]
    text_attention_mask = text['attention_mask'][:,:max_length-1]

    el_input_ids = event_labs['input_ids'][:,1:]
    el_attention_mask = event_labs['attention_mask'][:,1:]



    fuse_input_ids = torch.cat((text_input_ids,el_input_ids),axis = -1)
    fuse_attention_mask  = torch.cat((text_attention_mask ,el_attention_mask ),axis = -1)

    if fuse_input_ids.shape[1] > 512:

        seq_ids = fuse_input_ids[:,[-1]]
        seq_mask = fuse_input_ids[:,[-1]]
        input_ids_cliped = fuse_input_ids[:,:512-1]
        attention_mask_cliped = fuse_input_ids[:,:512-1]
        fuse_input_ids = torch.cat([input_ids_cliped,seq_ids],dim=-1)
        fuse_attention_mask = torch.cat([attention_mask_cliped,seq_mask],dim=-1)


    return {'input_ids':fuse_input_ids,
        'attention_mask': fuse_attention_mask}




def get_kmeans_centers(all_embeddings, num_classes):
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    # print(clustering_model.predict(all_embeddings))
    joblib.dump(clustering_model, 'weights/kmeans.model')

    return clustering_model.cluster_centers_



def collate_fn(data):    
    text_list = [d[0][0] for d in data]
    label_list = [d[1] for d in data]
    event_lab_list = [d[2][0] for d in data]
    return text_list,label_list,event_lab_list


def fit(model,label_tokens,dataloader):
    global Best_F1,Best_Roc    

    device = device1
    model.eval()
    model.to(device)
    embed_list = []

    for i,(text_list,label_list,event_list) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
    
            text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
            event_lab = tokenizer(event_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
    
            fuse_input =  cat_feature(max_length,text,event_lab)
            if text['input_ids'].shape[1] > max_length:
                text = clip_text(BATCH_SIZE,max_length,text,device)
            elif text['input_ids'].shape[1] < max_length:
                text = padding_text(BATCH_SIZE,max_length,text,device)
            embedding = model(fuse_input)
            embed_list.append(embedding)

    embed_list = torch.cat(embed_list,0).cpu()

    cluster_centers = torch.tensor(get_kmeans_centers(embed_list,number_cluster))

    torch.save(cluster_centers, f"xx.pth")


if __name__ == '__main__':

    train_dataset = PatientDataset(f'xx',class_3 = class_3,visit = visit,flag="all")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    train_length = train_dataset.__len__()
    print(train_length)

    model = mmdp()

    model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device1)), strict=False)

    fit(model,label_tokens,trainloader)








