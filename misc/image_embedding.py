#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn as nn
from torch.autograd import Variable

class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, feature_type='VGG'):
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
    
    def forward(self, input_image):
        input_image = input_image.view(-1, self.img_features)
        input_image = self.tanh(self.linear(input_image))
        
        return self.Dropout(input_image)
            

            


# In[ ]:




