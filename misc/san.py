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
                
    def forward(self, ques_feat, img_feat):
        # N: batch_size
        # stage1 #
        # image: N,m,d -> N,m,k
        # question: N,d -> N,k -> N,1,k
        #print("image feature shape",img_feat.shape)
        Wiavi1 = self.W_ia1(img_feat)
        Wqavq1 = self.W_qa1(ques_feat).unsqueeze(1)
        # ha1: N,m,k
        #print("image shape:",Wiavi1.shape)
        #print("question: shape:",Wqavq1.shape)
        ha1 = self.tanh(Wiavi1 + Wqavq1)
        # pi1: N,m,k -> N,m,1 -> N,m
        pi1 = self.softmax(self.Wp1(ha1).squeeze(2))
        # img_feat: N,m,d
        # pi1: N,m -> N,m,1
        # vI1: N,m,1 -> N,m
        vI1 = (img_feat*pi1.unsqueeze(2)).sum(dim = 1)
        u1 = vI1 + ques_feat
        
        # stage2 #
        # image: N,m,d -> N,m,k
        # question: N,d -> N,k -> N,1,k
        Wiavi2 = self.W_ia2(img_feat)
        Wqavq2 = self.W_qa2(ques_feat).unsqueeze(1)
        # ha2: N,m,k
        ha2 = self.tanh(Wiavi2 + Wqavq2)
        # pi1: N,m,k -> N,m,1 -> N,m
        pi2 = self.softmax(self.Wp2(ha2).squeeze(2))
        # img_feat: N,m,d
        # pi2: N,m -> N,m,1
        # vI2: N,m,d -> N,d
        vI2 = (img_feat*pi2.unsqueeze(2)).sum(dim = 1)
        u2 = vI2 + u1
        
        # N,d-> N,output_size
        score = self.fc(u2)
        
        
        # output_size = 1001
        return score
        
        
        
        
        
        
        
        
        

                
                
    


# In[ ]:




