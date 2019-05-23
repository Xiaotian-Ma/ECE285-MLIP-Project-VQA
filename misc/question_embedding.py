#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# In[37]:


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, use_gpu, bidirectional_or_not = False,num_layers=1, dropout=0.5):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()
        # 是否使用gpu #
        self.use_gpu = use_gpu
        # 我们用question里面用到的所有vocab的数量，来给我们提取出来的array再做embedding 
        # 这样得到的B*W*vocab_size(B个问题，每个问题W个单词，每个单词变成一个ocab_size维的向量)
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        # 进LSTM之前那个We
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional_or_not
        if self.bidirectional==False:
            self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers= num_layers, 
                            bias=True, batch_first=True, dropout=dropout,
                            bidirectional= self.bidirectional)
        else:
            self.LSTM = nn.LSTM(input_size=emb_size, hidden_size = 2*hidden_size, num_layers= num_layers, 
                            bias=True, batch_first=True, dropout=dropout,
                            bidirectional= self.bidirectional)
            
        
        
    
    
    def forward(self, ques_vec, ques_len): 
        B, W = ques_vec.size()
        
        #B*W -> B*W*(vocab_size+1)
        one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2,ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)
        # To remove additional column in one_hot, use slicing
        one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
        
        if self.use_gpu and torch.cuda.is_available():
            one_hot_vec = one_hot_vec.cuda()
        x = self.lookuptable(one_hot_vec)
        
        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.tanh(x)
        
        # h: [batch_size or B, 26 or W, hidden_size]
        # h: B * W * hidden_size
        # h_n: 各个层的最后一个时步的隐含状态h
        # c_n: 各个层的最后一个时步的隐含状态C: 1*batch*hidden_size
        output,(h_n,c_n)= self.LSTM(emb_vec)
        
        # 生成长度为question_len - 1的向量x #
        # 下面这一段同没看懂 先注释起来吧#
        '''x = torch.LongTensor(ques_len - 1)
        mask = torch.zeros(B, W).scatter_(1, x.view(-1, 1), 1)
        mask = Variable(mask.view(B, W, 1), requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            mask = mask.cuda()'''
        
        # output: [B, hidden_size]
        h_n = h_n.view(B,-1)
        
        return h_n
            
        
        


# In[38]:


#egmodel = QuestionEmbedding(100,30,40,0)
#ques_vec = torch.ones(10,18)
#ques_len = 10
#hn = egmodel.forward(ques_vec, ques_len )


# In[39]:


#hn.shape


# In[33]:


#h_n.shape


# In[34]:


#c_n.shape


# In[3]:


#print("ok")


# In[17]:


'''vocab_size = 100
ques_vec = torch.ones(30,18)
B, W = ques_vec.size()
one_hot_vec = torch.zeros(B, W, vocab_size+1).scatter_(2, ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)
print(one_hot_vec.shape)
one_hot_vec'''


# In[18]:


#B = 


# In[ ]:




