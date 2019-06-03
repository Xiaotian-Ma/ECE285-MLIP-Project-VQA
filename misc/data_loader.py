#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import torch
import numpy as np
#import utils as utils
from utils import *
import json

# ### 更改1：对答案做one——hot encoding

# In[4]:


class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, opt, train, transform=None, quiet=False):
        if not quiet:
            print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])
        
        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        # train = 1代表训练模式
        if train == 1:
            if not quiet:
                print('DataLoader loading h5 image train file: ' + opt['img_feature_path'])
            self.img_feature_path = opt['img_feature_path']
            self.ques = h5_file['/ques_train'][:]
            self.ans = h5_file['/ans_train'][:]
            self.split = h5_file['/split_train'][:]
            
        else:
            if not quiet:
                print('DataLoader loading h5 image test file: ' + opt['img_feature_path'])
            self.img_feature_path = opt['img_feature_path']
            self.ques = h5_file['/ques_val'][:]
            self.ans = h5_file['/ans_val'][:]
            self.split = h5_file['/split_val'][:]
        h5_file.close()
        
        self.feature_type = opt['feature_type']
        self.train = train
        self.transform = transform
        
        if not quiet:
            print('DataLoader loading json file: %s'% opt['json_file'])
            
        #json_file = utils.read_json(opt['json_file'])
        json_file = json.load(open(opt['json_file'],'r'))
        self.ix_to_word = json_file['ix_to_word']
        self.ix_to_ans = json_file['ix_to_ans']
        
        print("type of self.ques", self.ques)
        
        #self.vocab_size = utils.count_key(self.ix_to_word)
        self.vocab_size = len(self.ix_to_word)
        self.seq_length = self.ques.shape[1]
                                 
        
        
        
    
    
    def __len__(self):
        return self.split.shape[0]
    
    def getVocabSize(self):
        return self.vocab_size
    
    
    def __getitem__(self, idx):
        # 提取出来这个idx对应的img_idx
        #img_idx = self.img_pos[idx] - 1
        if self.img_feature_path:
            if self.train == 1:
                if self.feature_type == 'VGG':
                    # 提取出来第idx个 维度为（14，14，512）的训练特征（train set)
                    img = np.load(self.img_feature_path + '/image'+str(idx)+'.npy').reshape(14,14,512)
                elif self.feature_type == 'Residual':
                    # 提取出来第idx个 维度为（14，14，2048）的训练特征(train set)
                    print("extracting train image feature from ")
                    #img = self.h5_img_file['/images_train'][img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
                else:
                    print("Error(train): feature type error")
                    
            else:
                if self.feature_type == 'VGG':
                    # 提取出来第idx个 维度为（14，14，512）的测试特征（val set)
                    img = np.load('/datasets/home/home-03/98/898/wel031/san_vqa/data/image_feature_after_cnn/' + 'image'+str(idx)+'.npy').reshape(14,14,512)
                elif self.feature_type == 'Residual':
                    # 提取出来第idx个 维度为（14，14，2048）的测试特征(val set)
                    print("extracting train image feature from ")
                else:
                    print("Error(val): feature type error")
                    
        # 提取array格式的问题（问题已经做好了embedding映射） #
        question = np.array(self.ques[idx], dtype=np.int32) 
        
        # 提取问题长度（虽然我一直没搞懂这玩意干啥用的）#
        #ques_len = self.ques_len[idx].astype(int)
        
        # 答案（需要对答案进行one-hot encoding，不然你怎么做cross entropy……）
        answer = np.array(self.ans[idx])
        if self.transform is not None:
            img = self.transform(img)
            question = self.transform(question)
            
        #格式: （图片矩阵，问题，问题长度，答案） 
        return (img, question, answer)
    
    def getVocabSize(self):
        return self.vocab_size

    def getSeqLength(self):
        return self.seq_length
    
                    
            


# In[ ]:




