# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
import utils
import torch.nn.functional as F

from IPython.core.debugger import Pdb


# ### input_dim = 1024 or 2048 for both image and question embedding output dimension. ###
# 

# ### out_dim = 1024 or 360. The code using 2048 in_dim has out_dim as 360 ###
# ### I think we should decide  whether to do dimension decomposition based on out embedding out_dim ###

# ### embedding out_dim num_layers or R = exact number needs to be tested. 5 or 10 in this example.

# In[ ]:


class MutanFusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        # linear transformation for image_embed
        # use dropout with probability of 0.5
        # use tanh as activation function
        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        
         # create modulelist for image_embed
        self.image_transformation_layers = nn.ModuleList(hv)
        
        # linear transformation for question_embed
        # use dropout with probability of 0.5
        # use tanh as activation function
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
            
        # create modulelist for question_embed
        self.ques_transformation_layers = nn.ModuleList(hq)
        
    def forward(self, ques_emb, img_emb):
        # size of img_emb in_dim
        batch_size = img_emb.size()[0]
        
        # forward for ouput
        x_mm = []
        for i in range(self.num_layers):
            # forward for image_embed first
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)
            
            # forward for question_embed next
            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            
            # element-wsie multiplication
            #print("x_hq:",x_hq.shape)
            #print("x_hv",x_hv.shape)
            x_mm.append(torch.mul(x_hq.unsqueeze(1), x_hv))
        
        # stack up along out_dim
        x_mm = torch.stack(x_mm, dim=1)
        # sum up along col and resize
        #print("shape after summing up:",x_mm.sum(1).sum(1).shape)
        x_mm = torch.mean(x_mm.sum(1),dim = 1).view(batch_size, self.out_dim)
        # tanh activation
        x_mm = F.tanh(x_mm)
        return x_mm

