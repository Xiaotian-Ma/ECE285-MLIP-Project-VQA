#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import json

# ### 这里用的是不带question_length的写法

# In[2]:


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, use_gpu, dict_json_path, bidirectional_or_not = False,num_layers=1, dropout=0.5):
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
            self.num_directions = 1
        else:
            self.LSTM = nn.LSTM(input_size=emb_size, hidden_size = 2*hidden_size, num_layers= num_layers, 
                            bias=True, batch_first=True, dropout=dropout,
                            bidirectional= self.bidirectional)
            self.num_directions = 2
            
        self.num_layers  = num_layers
        self.hidden_size = hidden_size 
        self.dict_json_path = dict_json_path
        
    
    
    #def forward(self, ques_vec, ques_len):
    def forward(self, ques_vec, method):
        if method == 1:   
        # B: batch_size
        # W: question的维度（这里处理为20 大概是每个句子最长包含几个单词）
            B, W = ques_vec.size()
        #B * W -> B * W * (vocab_size+1)
            one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2,ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)
        # To remove additional column in one_hot, use slicing
            one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
            if self.use_gpu and torch.cuda.is_available():
                one_hot_vec = one_hot_vec.cuda()   
        # 做一层线性变换: batch_size * W * vocab_size --> batch_size * W * emb_size #
            x = self.lookuptable(one_hot_vec)
        # emb_vec: batch_size * W * emb_size
            emb_vec = self.tanh(x)
            
        
        if method ==2:
            B, W = ques_vec.size()
            self.word_dict = json.load(open(self.dict_json_path))['word_to_ix']
            self.word_vocab_size = len(self.word_dict)
            self.word_emb = nn.Embedding(self.vocab_size, self.emb_size)
            
            if self.use_gpu and torch.cuda.is_available():
                self.word_emb = self.word_emb.cuda()
                
            ques_vec = ques_vec.long()
            sentence_index = Variable(ques_vec,requires_grad=False)
            #print("sentence_index:",sentence_index)
            sentence_embed = self.word_emb(sentence_index)
            emb_vec = self.tanh(sentence_embed)
            
            
        final_output = torch.zeros(B,self.hidden_size)
        final_output = final_output.cuda()
        for i in range(emb_vec.shape[0]):
            #print("self.num_layers:", self.num_layers)
            #print("self.num_directions:", self.num_directions)
            h_0 = torch.zeros(self.num_layers*self.num_directions , 1, self.hidden_size)
            c_0 = torch.zeros(self.num_layers*self.num_directions , 1, self.hidden_size) 
            h_0, c_0 = h_0.cuda(), c_0.cuda()
            for j in range(emb_vec.shape[1]):
                x_t = emb_vec[i][j].unsqueeze(0)
                x_t = x_t.unsqueeze(0)
                #print("input h_0 shape:",h_0.shape)
                #print("input c_0 shape:",c_0.shape)
                output_t,(h_t,c_t) = self.LSTM(x_t,(h_0,c_0))
                h_0 = h_t
                c_0 = c_t
            #print("output shape: ",h_t.shape)
            final_output[i] = h_t.squeeze(0).squeeze(0)
    
        return final_output
    
    
    
    '''def forward_with_embedding_matrix(self, dict_json_path, ques_vec):
        
        B, W = ques_vec.size()
        self.word_dict = json.load(open('dict_json_path'))['word_to_ix']
        self.word_vocab_size = len(json.load(open('dict_json_path'))['word_to_ix'])
        self.word_emb = nn.Embedding(self.vocab_size, emb_size)
        
        
        sentence_index = Variable(ques_vec,requires_grad=False)
        sentence_embed = self.word_emb(sentence_index)
        
        emb_vec = self.tanh(sentence_embed)
        print("minibatch_number:",emb_vec.shape[0])
        
        final_output = torch.zeros(B,self.hidden_size)
        for i in range(emb_vec.shape[0]):
            h_0 = torch.zeros(self.num_layers*self.num_directions , 1, self.hidden_size)
            c_0 = torch.zeros(self.num_layers*self.num_directions , 1, self.hidden_size)
            for j in range(emb_vec.shape[1]):
                x_t = emb_vec[i][j].unsqueeze(0)
                x_t = x_t.unsqueeze(0)
                output_t,(h_t,c_t) = self.LSTM(x_t,(h_0,c_0))
                h_0 = h_t
                c_0 = c_t
            final_output[i] = h_t.squeeze(0).squeeze(0)
        
        return final_output'''
            
        





