#!/usr/bin/env python
# coding: utf-8

# In[3]:

import torch
import torch.nn as nn
from torch.autograd import Variable

class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, use_gpu, feature_type='VGG'):
        super(ImageEmbedding, self).__init__()
        
        if feature_type == 'VGG':
            self.img_features = 512
        elif feature_type == 'Residual':
            self.img_features = 2048
        else:
            print('Unsupported feature type: \'{}\''.format(feature_type))
        
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.img_features, self.hidden_size)
        self.tanh = nn.Tanh()
        self.Dropout = nn.Dropout(0.5)
        self.use_gpu = use_gpu
    
    def forward(self, input_image):
        #if self.use_gpu and torch.cuda.is_available():
            #input_image = input_image.cuda()
            
        input_image = input_image.view(-1, self.img_features)
        input_image = self.tanh(self.linear(input_image))
        
        return self.Dropout(input_image)
            

            


# In[2]:


#import torch
#import torch.nn as nn


# In[12]:


'''image1 = torch.ones(1,512,14,14)
img_features = 512
linear = nn.Linear(512,600)
input_image = image1.view(-1, img_features)
print(input_image.shape)
linear(input_image).shape'''


# In[10]:


'''example1 = torch.ones(3,10)
linear1 = nn.Linear(10,4)
print(example1.shape)
print(linear1)'''


# In[11]:


#linear1(example1).shape


# In[3]:


#layer1=nn.Softmax()


# In[6]:


#softmaxeg1 = torch.ones(1,2)
#softmaxeg1 = softmaxeg1.transpose(0,1)
#print(softmaxeg1.shape)
#layer1(softmaxeg1)


# In[ ]:




