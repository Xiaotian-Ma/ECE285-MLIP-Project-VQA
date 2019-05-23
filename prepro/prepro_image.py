#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import json


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[27]:


train_json_path = '/datasets/home/home-03/98/898/wel031/san_vqa/vqa_raw_train.json'
val_json_path = '/datasets/home/home-03/98/898/wel031/san_vqa/vqa_raw_test.json'


# In[3]:


vgg16model = tv.models.vgg16_bn(pretrained = True)


# In[28]:


class ImageDataset(td.Dataset):
    def __init__(self, train_json_path, val_json_path, image_size=(448, 448)):
        super(ImageDataset, self).__init__()
        self.image_size = image_size
        self.model = vgg16model.features
        
        train_json = json.load(open(train_json_path, 'r'))
        val_json = json.load(open(val_json_path, 'r'))
        
        self.train_path_list = ['/datasets/ee285f-public/VQA2017/'+ w['img_path']+'.jpg' for w in train_json]
        self.val_path_list = ['/datasets/ee285f-public/VQA2017/'+ w['img_path']+'.jpg' for w in val_json]
                          
    def number_of_train_images(self):
        return len(self.train_path_list)
    
    def number_of_val_images(self):
        return len(self.val_path_list)
    
    def forward(self, start_number, end_number, mode):
        transform = tv.transforms.Compose([tv.transforms.Resize((448,448)),tv.transforms.ToTensor(),tv.transforms.Normalize(mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5))])
        
        if mode =='train':
            final_output = torch.ones(end_number-start_number+1 ,512, 14, 14)
            for i in range(start_number,end_number):
                img = Image.open(self.train_path_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1,3,448,448)
                final_output[i] = self.model(img)
                
        if mode =='val':
            final_output = torch.ones(end_number-start_number+1 ,512, 14, 14)
            for i in range(start_number,end_number):
                img = Image.open(self.val_path_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1,3,448,448)
                final_output[i] = self.model(img)
                
        return final_output
                
            
        


# In[31]:


#egdataset = ImageDataset(train_json_path, val_json_path)


# In[32]:


'''eg =  egdataset.forward(1,2,'train')
eg.shape'''


# In[ ]:




