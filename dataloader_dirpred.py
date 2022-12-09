import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
import random
from datetime import datetime
from collections import defaultdict

SEED = 2019
torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True,TOKENIZERS_PARALLELISM=True)

class PatientDataset(object):
    def __init__(self, data_dir,class_3,visit,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.event_dir = 'dataset/event/'

        self.text_dir = 'dataset/brief_course/'
        self.numeric_dir = 'dataset/alldata/all/'
        self.stopword = list(pd.read_csv('stopwods.csv').values.squeeze())
        self.visit = visit
        self.low = [2.80000000e+01, -7.50000000e-02,  4.30000000e+01, 4.00000000e+01,
                    4.10000000e+01,  9.00000000e+01,  5.50000000e+00,  6.15000000e+01,  
                    3.50000000e+01,  3.12996266e+01, 7.14500000e+00] 
        self.up = [  92.,           0.685,         187.,         128.,   
                    113.,         106.,          33.5,        177.5,         
                    38.55555556, 127.94021917,   7.585]
        self.sbj_dir = os.path.join(f'{data_dir}',flag)
        self.sbj_list = os.listdir(self.sbj_dir)

        self.max_length = 1000
        self.class_3 = class_3
        self.feature_list = [
        'Diastolic blood pressure',
        'Fraction inspired oxygen', 
        'Glucose', 
        'Heart Rate', 
        'Mean blood pressure', 
        'Oxygen saturation', 
        'Respiratory rate',
        'Systolic blood pressure', 
        'Temperature', 
        'Weight', 
        'pH']
        self.label_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
    def padding_text(self,vec):
        input_ids = vec['input_ids']
        attention_mask = vec['attention_mask']
        padding_input_ids = torch.ones((input_ids.shape[0],self.max_length-input_ids.shape[1]),dtype = int).to(self.device)
        padding_attention_mask = torch.zeros((attention_mask.shape[0],self.max_length-attention_mask.shape[1]),dtype = int).to(self.device)
        input_ids_pad = torch.cat([input_ids,padding_input_ids],dim=-1)
        attention_mask_pad = torch.cat([attention_mask,padding_attention_mask],dim=-1)
        vec = {'input_ids': input_ids_pad,
        'attention_mask': attention_mask_pad}
        return vec
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)

        return temp
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def __getitem__(self, idx):
    
        patient_id = self.sbj_list[idx]
        visit_list = sorted(os.listdir(os.path.join(self.data_dir,self.flag, patient_id)), key= self.sort_key)
        label_list = []
        breif_course_list = []
        event_lab_list = []
        event_lab_length = []
        numeric_data = None

        for patient_file in visit_list:
            event_exist = True
            lab_exist = True
            text_df = pd.read_csv(self.text_dir+"_".join(patient_file.split("_")[:2])+".csv").values
            breif_course = text_df[:,1:2].tolist()
            breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
            text = ' '.join(breif_course)
            text = self.rm_stop_words(text)
            text_length = len(tokenizer.tokenize(text))

            breif_course_list.append(text)

            numeric_data_file =  self.numeric_dir + patient_file.split("_")[0] + "_" + patient_file.split("_")[2].replace("eposide","episode").strip(".csv") + "_timeseries.csv"
            lab_dic = defaultdict(list)

            if os.path.exists(numeric_data_file):

                numeric_data = pd.read_csv(numeric_data_file)[self.feature_list].values
                for l in range(numeric_data.shape[-1]):
                    if (np.array(numeric_data[:,l]) < self.low[l]).any():
                            lab_dic[l].append("low")
                    elif (np.array(numeric_data[:,l]) > self.up[l]).any():
                            lab_dic[l].append("high")
                    else:
                        lab_dic[l].append("normal")
            else:
                lab_exist = False

            # print(lab_dic)
            lab_description = []
            for k in lab_dic.keys():
                strs =  str(lab_dic[k][0]) + " " + str(self.feature_list[k]) 
                lab_description.append(strs.lower())

            event_codes = []
            if not os.path.exists(os.path.join(self.event_dir, patient_file.split("_")[0] + "_" + patient_file.split("_")[2].replace("eposide","episode").strip(".csv") + "_timeseries.csv")):
                event_exist = False
            else:
                event_file = pd.read_csv(os.path.join(self.event_dir, patient_file.split("_")[0] + "_" + patient_file.split("_")[2].replace("eposide","episode").strip(".csv") + "_timeseries.csv"))[["procedure_event","input_event_mv","input_event_cv"]].values
            
                for i in range((len(event_file))):
                    e = event_file[i]
                    for j in e:
                        if not pd.isnull(j):
                            j = j.lower()
                            j = re.sub(r'[^a-zA-Z\s]', '', j)
                            if j in event_codes: continue
                            if j == 'ekg': 
                                j ="electrocardiogram "
                            if 'or ' in j: 
                                j = j.replace('or ',"")
                            if j == " gauge": j = "gauge"
                            if len(j)<3: continue
                            event_codes.append(j)


            if len(event_codes):
                while True:
                    if '' not in event_codes: break
                    event_codes.remove('')
                
                event_lab = 'clinical events include ' +  " ".join(event_codes)
                event_length =  len(tokenizer.tokenize('clinical events include ' + ' '.join(event_codes).replace("  "," ")))
                if lab_exist:
                    event_lab = 'clinical events include ' + ' '.join(event_codes)  + " lab " + ' '.join(lab_description) 
                else:
                    event_lab = 'clinical events include ' + ' '.join(event_codes)  + " " + 'not lab'

            else:
                event_lab = 'not includes clincial events'
                event_length = 0
                if lab_exist:
                    event_lab = 'not includes clincial events' + ' '.join(event_codes)  + " lab " + ' '.join(lab_description) 
                else:
                    event_lab = 'not includes clincial events' + ' '.join(event_codes)  + " " + 'not lab'
            event_lab = event_lab.replace("  "," ")
            event_lab_length.append((text_length,event_length))
            event_lab_list.append(event_lab)
            if self.visit == 'twice':
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag+"1",patient_file))[self.label_list].values[:1,:][0])

            else:
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag,patient_file))[self.label_list].values[:1,:][0])

            cluster_label = [0,0,0]
            if self.class_3:
                if sum(label[:13]) >=1:
                    cluster_label[0] = 1
                if sum(label[13:20]) >= 1:
                    cluster_label[1] = 1
                if sum(label[20:]) >= 1:
                    cluster_label[2] = 1
                label_list.append(cluster_label)
            else:
                label_list.append(label)
        return breif_course_list,label_list, event_lab_list,event_lab_length


    def __len__(self):
        return len(self.sbj_list)


def collate_fn(data):    
    text_list = [d[0] for d in data]
    label_list = [d[1] for d in data]
    event_lab_list = [d[2] for d in data] 
    return text_list,label_list,event_lab_list


