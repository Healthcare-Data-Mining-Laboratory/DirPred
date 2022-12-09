import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_dirpred import PatientDataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils import clip_grad_norm_
from model_dirpred import mmdp
from sklearn.metrics import *
import copy
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"

SEED = 2019 
# SEED = 42 
# SEED = 1069
# SEED = 1999 
# SEED = 3248 



torch.manual_seed(SEED)

## scigpu10  dynamic_dirichlet


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
num_epochs = 100
max_length = 300
BATCH_SIZE = 3
latent_ndims = 30 
clip_value = 500
prompt_tokens =  5
bert_max_length = 512 - prompt_tokens*3
# 8, 10, 16, 30, 40, 50, 80

CLIP = True
class_3 = True

evaluation = False
pretrained = True
Freeze = True
SV_WEIGHTS = True
logs = True
visit = 'twice'

weight_dir = "xxx"


#gpu 13


start_epoch = 0

Best_Roc = 0.82
Best_F1 = 0.890

save_dir = "xx"
save_name = f"xx"
log_file_name = f'xx.txt'

prior_alpha = torch.Tensor([1.])

device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)



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
    sentence_difference = max_length - len(input_ids[0])
    padding_ids = torch.ones((1,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((1,sentence_difference), dtype = torch.long).to(device)

    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec

def collate_fn(data):    
    text_list = [d[0] for d in data]
    label_list = [d[1] for d in data]
    event_lab_list = [d[2] for d in data] 
    length_list = [d[3] for d in data] 
    return text_list,label_list,event_lab_list,length_list


def _cat_learned_embedding_to_input(model,input_ids,length_list,feature) -> torch.Tensor:
    inputs_embeds = model.text_encoder.embeddings.word_embeddings(input_ids)

    if len(list(inputs_embeds.shape)) == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)

    # [batch_size, n_tokens, n_embd]
    if feature == "text":
        learned_embeds = model.soft_text_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
    elif feature == "event":
        learned_embeds = model.soft_event_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

    elif feature == "label":
        learned_embeds = model.soft_label_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
    elif feature == "fuse":
        learned_embeds_text = model.soft_text_prompt.weight.unsqueeze(0)
        learned_embeds_event = model.soft_event_prompt.weight.unsqueeze(0)
        learned_embeds_lab = model.soft_lab_prompt.weight.unsqueeze(0)
        inputs_embed_list = []
        event_position = length_list[0]
        if event_position > max_length - 1:
                event_position = max_length - 1
        lab_position =  length_list[1]

        inputs_embeds = torch.cat([learned_embeds_text, inputs_embeds[:,:event_position,:],learned_embeds_event,inputs_embeds[:,event_position:event_position+lab_position,:],learned_embeds_lab,inputs_embeds[:,event_position+lab_position:,:]], dim=1)
        return inputs_embeds
    inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

    return inputs_embeds
def _extend_attention_mask(n_tokens,attention_mask,length_list,feature = "text"):
    if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
    n_batches = attention_mask.shape[0]

    if feature == 'text':
       
        return torch.cat([torch.full((n_batches, n_tokens), 1).to(attention_mask.device), attention_mask],dim = 1,)
    else:
        event_position = length_list[0]
        if event_position > max_length - 1: event_position = max_length - 1
        lab_position =  length_list[1]
        single_attention_mask = torch.cat([torch.full((1, n_tokens), 1).to(attention_mask.device), attention_mask[:,:event_position],torch.full((1, n_tokens), 1).to(attention_mask.device),attention_mask[:,event_position:event_position+lab_position],torch.full((1, n_tokens), 1).to(attention_mask.device),attention_mask[:,event_position+lab_position:]],dim = 1,)
        return single_attention_mask


def cat_feature(max_length,text,event_labs):
    text_input_ids = text['input_ids'][:,:max_length-1]
    text_attention_mask = text['attention_mask'][:,:max_length-1]

    el_input_ids = event_labs['input_ids'][:,1:]
    el_attention_mask = event_labs['attention_mask'][:,1:]



    fuse_input_ids = torch.cat((text_input_ids,el_input_ids),axis = -1)
    fuse_attention_mask  = torch.cat((text_attention_mask ,el_attention_mask ),axis = -1)

    if fuse_input_ids.shape[1] > bert_max_length:

        seq_ids = fuse_input_ids[:,[-1]]
        seq_mask = fuse_input_ids[:,[-1]]
        input_ids_cliped = fuse_input_ids[:,:bert_max_length-1]
        attention_mask_cliped = fuse_input_ids[:,:bert_max_length-1]
        fuse_input_ids = torch.cat([input_ids_cliped,seq_ids],dim=-1)
        fuse_attention_mask = torch.cat([attention_mask_cliped,seq_mask],dim=-1)


    return {'input_ids':fuse_input_ids,
        'attention_mask': fuse_attention_mask}



    return text_list,label_list,event_lab_list,length_list

def logistic_func(x):
    return 1 / (1 + torch.exp(-x))
def beta_func(a, b):
    return (torch.lgamma(a) + torch.lgamma(b)-torch.lgamma(a+b)).exp()

def kl_divergence(param1, param2,prior_alpha,prior_beta):
    # compute taylor expansion for E[log (1-v)] term                                                                                                                                             
    # hard-code so we don't have to use Scan()                                                                                                                                                   
    kl = 1./(1+param1*param2) * beta_func(1./param1, param2)
    kl += 1./(2+param1*param2) * beta_func(2./param1, param2)
    kl += 1./(3+param1*param2) * beta_func(3./param1, param2)
    kl += 1./(4+param1*param2) * beta_func(4./param1, param2)
    kl += 1./(5+param1*param2) * beta_func(5./param1, param2)
    kl += 1./(6+param1*param2) * beta_func(6./param1, param2)
    kl += 1./(7+param1*param2) * beta_func(7./param1, param2)
    kl += 1./(8+param1*param2) * beta_func(8./param1, param2)
    kl += 1./(9+param1*param2) * beta_func(9./param1, param2)
    kl += 1./(10+param1*param2) * beta_func(10./param1, param2)
    kl *= (prior_beta-1)*param2

    # use another taylor approx for Digamma function                                                                                                                                             
    psi_b_taylor_approx = (param2).log() - 1./(2 * param2) - 1./(12 * param2**2)

    kl += (param1-prior_alpha)/param1 * (-0.57721 - psi_b_taylor_approx - 1/param2) #T.psi(self.posterior_b)                                                                                        

    # add normalization constants                                                                                                                                                                
    kl += (param1*param2).log() + (beta_func(prior_alpha, prior_beta)).log()

    # final term                                                                                                                                                                                 
    kl += -(param2-1)/param2
    # print(param1,param2)

    # print("loss: ",kl.sum())

    return kl.sum()

def fit(epoch,model,center_embedding,label_tokens,y_bce_loss,dist_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc,prior_alpha,log_file_name
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    dist_loss.to(device)
    prior_alpha = prior_alpha.to(device)
    center_embedding = torch.nn.Parameter(center_embedding).to(device)
    uniform_distribution = torch.distributions.uniform.Uniform(low=torch.Tensor([.01]).to(device), high=torch.Tensor([.99]).to(device))

    eopch_loss_list = []
    epoch_classify_loss_list = []
    epoch_sb_kl_loss_list = []
    y_list = []
    pred_list_f1 = []
    pred_list_roc = []
    cluster_id_list = []
    embedding_list = []


    for i,(text_list,label_list,event_lab_list,position_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        # if i == 20:
        #     break
        batch_sbkl_list = torch.zeros(len(text_list)).to(device)
        batch_cls_list = torch.zeros(len(text_list)).to(device)
        if flag == "train":
    
            with torch.set_grad_enabled(True):
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_lab_event = event_lab_list[p]
                    p_label = label_list[p]
                    p_length = position_list[p]

                    Ztd_zero = torch.randn((1, 384)).to(device)
                    Ztd_zero.requires_grad = True
                    sbkl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)
                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]
                    for v in range(len(p_lab_event)):
                        text = p_text[v]
                        event_lab = p_lab_event[v]
                        position = p_length[v]
                        label = p_label[v]
                        label = torch.tensor(label).to(torch.float32).to(device)

                        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        event_lab = tokenizer(event_lab, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        fuse_input =  cat_feature(max_length,text,event_lab)

                        fuse_input = {
                        'inputs_embeds': _cat_learned_embedding_to_input(model,fuse_input['input_ids'],position,'fuse'),
                            'attention_mask':  _extend_attention_mask(prompt_tokens,fuse_input['attention_mask'],position,'fuse').to(device)
                        }  


                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)

                     
                        if v == 0:
                            Ztd_last = Ztd_zero
                        pred,alpha,beta,prior_beta,u,z,pi  = \
                        model(Ztd_list,Ztd_last,fuse_input,center_embedding,uniform_distribution)            
                        pi = pi.squeeze().cpu().data.numpy().tolist()
                        cluster_id = pi.index(max(pi))+1
                        cluster_id_list.append(cluster_id)   
                        embedding_list.append(z.squeeze().cpu().data.tolist())
                        Ztd_last = z 
                        Ztd_list.append(z)
                        s_cls =  y_bce_loss(pred.squeeze(),label.squeeze())
                        s_sbkl = kl_divergence(alpha,beta,prior_alpha,prior_beta)
                        sbkl_loss[v] = s_sbkl
                        cls_loss[v] = s_cls
                        label = np.array(label.cpu().data.tolist())
                        pred = np.array(pred.cpu().data.tolist())
                        pred_list_roc.append(pred)
                        pred=(pred > 0.5)*1

                        y_list.append(label)
                        pred_list_f1.append(pred)
                    # print(model.soft_label_prompt.weight.squeeze())

                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = sbkl_loss.view(-1).mean()

                    batch_cls_list[p] = cls_loss_p
                    batch_sbkl_list[p] = kl_loss_p

                batch_sbkl_list = batch_sbkl_list.view(-1).mean()
                batch_cls_list = batch_cls_list.view(-1).mean()
                total_loss = batch_cls_list + batch_sbkl_list/latent_ndims
                total_loss.backward(retain_graph=True)
                if CLIP:
                    clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()

                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_sb_kl_loss_list.append(batch_sbkl_list.cpu().data)
   
        else:
            with torch.no_grad():
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_lab_event = event_lab_list[p]
                    p_label = label_list[p]
                    p_length = position_list[p]

                    Ztd_zero = torch.randn((1, 384)).to(device)
                    Ztd_zero.requires_grad = True
                    sbkl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)
                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]
                    for v in range(len(p_lab_event)):
                        text = p_text[v]
                        event_lab = p_lab_event[v]
                        position = p_length[v]
                        label = p_label[v]
                        label = torch.tensor(label).to(torch.float32).to(device)

                        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        event_lab = tokenizer(event_lab, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        fuse_input =  cat_feature(max_length,text,event_lab)

                        fuse_input = {
                        'inputs_embeds': _cat_learned_embedding_to_input(model,fuse_input['input_ids'],position,'fuse'),
                            'attention_mask':  _extend_attention_mask(prompt_tokens,fuse_input['attention_mask'],position,'fuse').to(device)
                        }  

                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)

                     
                        if v == 0:
                            Ztd_last = Ztd_zero
                        pred,alpha,beta,prior_beta,u,z,pi  = \
                        model(Ztd_list,Ztd_last,fuse_input,center_embedding,uniform_distribution)            
                        pi = pi.squeeze().cpu().data.numpy().tolist()
                        cluster_id = pi.index(max(pi))+1
                        # if cluster_id in [10,30,18,8,11]:continue

                        cluster_id_list.append(cluster_id)  
                        embedding_list.append(z.squeeze().cpu().data.tolist())
 
                        Ztd_last = z 
                        Ztd_list.append(z)
                        s_cls =  y_bce_loss(pred.squeeze(),label.squeeze())
                        s_sbkl = kl_divergence(alpha,beta,prior_alpha,prior_beta)
                        sbkl_loss[v] = s_sbkl
                        cls_loss[v] = s_cls
                        label = np.array(label.cpu().data.tolist())
                        pred = np.array(pred.cpu().data.tolist())
                        pred_list_roc.append(pred)
                        pred=(pred > 0.5)*1

                        y_list.append(label)
                        pred_list_f1.append(pred)

                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = sbkl_loss.view(-1).mean()

                    batch_cls_list[p] = cls_loss_p
                    batch_sbkl_list[p] = kl_loss_p

                batch_sbkl_list = batch_sbkl_list.view(-1).mean()
                batch_cls_list = batch_cls_list.view(-1).mean()
                total_loss = batch_cls_list + batch_sbkl_list/latent_ndims
                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_sb_kl_loss_list.append(batch_sbkl_list.cpu().data)
    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)
    pred_list_roc = np.vstack(pred_list_roc)
    acc = metrics.accuracy_score(y_list,pred_list_f1)

    # [(3, 393), (1, 151), (2, 120), (5, 110), (10, 45), (7, 33), (30, 32), (6, 30), (4, 17), (18, 11), (8, 1), (12, 1)]
    cluster_id_list = np.array(cluster_id_list)
    embedding_list =  np.array(embedding_list)
    shi_score = silhouette_score(embedding_list, cluster_id_list)
    db_score = davies_bouldin_score(embedding_list, cluster_id_list)
    vic_score = calinski_harabasz_score(embedding_list, cluster_id_list)
    label_count = Counter(cluster_id_list).most_common()
    print(label_count)


    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_roc,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_roc,average="macro")
    total_loss = sum(eopch_loss_list) / len(eopch_loss_list)
    total_cls_loss = sum(epoch_classify_loss_list) / len(epoch_classify_loss_list)
    total_sb_kl_loss = sum(epoch_sb_kl_loss_list) / len(epoch_sb_kl_loss_list)
    
   
    print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC : {} | ACC: {} | SHI Score: {} |  DB Score: {} | VIC Score: {} | Total CLS LOSS  : {} | Total CLUSTER LOSS  : {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,acc,shi_score,db_score,vic_score,total_cls_loss, total_sb_kl_loss,total_loss))
   
    if flag == 'test':
        if logs:
            with open(f'{log_file_name}.txt', 'a+') as log_file:
                log_file.write("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC : {} | ACC: {} | SHI Score: {} |  DB Score: {} | VIC Score: {} | Total CLS LOSS  : {} | Total CLUSTER LOSS  : {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,acc,shi_score,db_score,vic_score,total_cls_loss, total_sb_kl_loss,total_loss)+'\n')
                log_file.close()
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(total_loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(total_loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro


if __name__ == '__main__':
    print("load center from ", f"xx.pth")
    center_embedding = torch.load(f"xx.pth").type(torch.cuda.FloatTensor)

    train_dataset = PatientDataset(f'xx',class_3 = class_3,visit = visit,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f'xx',class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model = mmdp(class_3,latent_ndims,prompt_tokens)

    if pretrained:
        print(f"loading weights: {weight_dir}")
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    ### freeze parameters ####
    # optimizer = optim.Adam(model.parameters(True), lr = 1e-5)
    ignored_params = list(map(id, model.text_encoder.parameters())) 

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters()) 
    optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.text_encoder.parameters(), 'lr': 1e-5}], 3e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i in range(0,1) :
                for param in child.parameters():
                    param.requires_grad = False
    ##########################


    y_bce_loss = nn.BCELoss()
    dist_loss = nn.CrossEntropyLoss()

    if evaluation:
        precision_micro_list = []
        precision_macro_list = []
        recall_micro_list = []
        recall_macro_list = []
        f1_micro_list = []
        f1_macro_list = []
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(1):
    
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,center_embedding,label_tokens,y_bce_loss,dist_loss,testloader,optimizer,flag='test')
  

    else:
        for epoch in range(start_epoch,num_epochs):
            # fit(epoch,model,text_recon_loss,y_bce_loss,trainloader,optimizer,flag='train')
            # fit(epoch,model,text_recon_loss,y_bce_loss,testloader,optimizer,flag='test')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,label_tokens,y_bce_loss,dist_loss,trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,label_tokens,y_bce_loss,dist_loss,testloader,optimizer,flag='test')



   

 







