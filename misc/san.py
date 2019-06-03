#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, input_size, att_size, img_seq_size, output_size, drop_ratio):
        super(Attention, self).__init__() 
        # d = input_size | m = img_seq_size | k = att_size
        # number of multiple answers = output_size
        self.W_ia1 = nn.Linear(input_size, att_size, bias = False)
        self.W_qa1 = nn.Linear(input_size, att_size, bias = True)
        self.Wp1 = nn.Linear(att_size, 1, bias = True)
        
        self.W_ia2 = nn.Linear(input_size, att_size, bias = False)
        self.W_qa2 = nn.Linear(input_size, att_size, bias = True)
        self.Wp2 = nn.Linear(att_size, 1, bias = True)
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
        
        self.fc = nn.Linear(input_size, output_size, bias=True)
        
        self.input_size = input_size
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size
        
    def forward(self, ques_feat, img_feat):
        # Stack 1
        # d = input_size | m = img_seq_size | k = att_size
        #img_feat: d * m -> m * d 
        #img: m * d * d* k -> m * k -> k * m
        img_feat = img_feat.transpose(0,1)
        W_iaV_i1 = self.W_ia1(img_feat).transpose(0,1)
        
        #question_feat: 1 * d(hidden output of last lstm layer)
        #question: 1 * d * d * k -> 1 * k -> k * 1
        W_qaV_q1 = self.W_qa1(ques_feat).transpose(0,1)
        
        # hA1: k * m -> m * k
        # Wp(hA1): m * k * k * 1 -> m * 1 -> 1 * m
        # p1 = softmax(WphA1)
        hA1 = self.tanh(W_iaV_i1 + W_qaV_q1.expand(self.att_size, self.img_seq_size)).transpose(0,1)
        WphA1 = self.Wp1(hA1).transpose(0,1)
        p1 = self.softmax(WphA1)
        
        #update vI and score
        #p1: 1 * m (adds up to 1)
        # vI: m * d(m vectors of 1 * d)
        # img_feat: vI(1 * d)
        p1T = p1.transpose(0,1)
        p1T = p1T.expand(self.img_seq_size, self.input_size)
        vI1 = p1T.mul(img_feat).sum(dim = 0)
        
        #u1: 1 * d (same as question_feat)
        u1 = vI1 + ques_feat
        
        
        
        # Stack 2
        #img: m * d * d* k -> m * k -> k * m
        W_iaV_i2 = self.W_ia2(img_feat).transpose(0,1)
        
        #question_feat: 1 * d(hidden output of last lstm layer)
        #question: 1 * d * d * k -> 1 * k -> k * 1
        W_qaV_q2 = self.W_qa2(u1).transpose(0,1)
        
        # d = input_size | m = img_seq_size | k = att_size
        # hA2: k * m -> m * k
        # Wp(hA2): m * k * k * 1 -> m * 1 -> 1 * m
        # p2 = softmax(WphA2)
        hA2 = self.tanh(W_iaV_i2 + W_qaV_q2.expand(self.att_size, self.img_seq_size)).transpose(0,1)
        WphA2 = self.Wp2(hA2).transpose(0,1)
        p2 = self.softmax(WphA2)
        
        #update vI and score
        #p2: 1 * m -> m * 1(adds up to 1)
        # vI2: m * d(m vectors of 1 * d)
        # img_feat: vI(1 * d)
        p2T = p2.transpose(0,1)
        p2T = p2T.expand(self.img_seq_size, self.input_size)
        vI2 = p2T.mul(img_feat).sum(dim = 0)
        
        #u2: 1 * d (same as question_feat)
        u2 = vI2 + u1
        
        #d = input_size: 
        # transformation: input_size -> output_size
        u2 = self.fc(u2)
        
        return u2
        
        
        
        
        
        
        
        
        

                
                
    


# In[ ]:




