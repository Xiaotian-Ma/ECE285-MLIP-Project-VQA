#!/usr/bin/env python
# coding: utf-8

# In[27]:


# ref: https://github.com/Shivanshu-Gupta/Visual-Question-Answering/blob/master/san.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils
from IPython.core.debugger import Pdb

class ImageEmbedding(nn.Module):
    def __init__(self, output_size, mode = 'train', extract_features = False, features_dir = None):
        super(ImageEmbedding, self).__init__()
        # use pretrained resNet152
        
        # cnn: output 2048 for per image
        self.cnn = nn.Sequential(models.resnet152(pretrained=True).conv1,
                                 models.resnet152(pretrained=True).bn1,
                                 models.resnet152(pretrained=True).relu,
                                 models.resnet152(pretrained=True).maxpool,
                                    models.resnet152(pretrained=True).layer1,
                                    models.resnet152(pretrained=True).layer2,
                                   models.resnet152(pretrained=True).layer3,
                                   models.resnet152(pretrained=True).layer4,
                                   models.resnet152(pretrained=True).avgpool)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        
            
        self.mode = mode
        self.extract_features = extract_features
        self.features_dir = features_dir
        
    def forward(self,image, image_ids):
        if not self.extract_features:
            # input: , ---->  output:2048
            image = self.cnn(image)
            
            # change shape #
            image = image.view(-1, 512, 196).transpose(1, 2)
            if self.features_dir is not None:
                utils.save_image_features(image, image_ids, self.features_dir)
        
        image_embedding = self.fc(image)
        return image_embedding
        
# this needs to be modified....maybe using Glove?        
class QuesEmbedding(nn.Module):
    def __init__(self, input_size=500, output_size=1024, num_layers=1, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size, batch_first=batch_first)

    def forward(self, ques):
        # seq_len * N * 500 -> (1 * N * 1024, 1 * N * 1024)
        _, hx = self.lstm(ques)
        
        # (1 * N * 1024, 1 * N * 1024) -> 1 * N * 1024
        h, _ = hx
        ques_embedding = h[0]
        return ques_embedding
 

class Attention(nn.Module):
    def __init__(self, d=2048, k=512, dropout=True):
        super(Attention, self).__init__()
        # Linear(in_features=2048, out_features=1000, bias=True) #
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        #改变最后一个维度 #
        # N * 196 * 2048 -> N * 196 * 512
        hi = self.ff_image(vi)
        
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha)
        
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(self, vocab_size, word_emb_size=500, emb_size=2048, att_ff_size=512, output_size=1000,
                 num_att_layers=1, num_mlp_layers=1, mode='train', extract_img_features=True, features_dir=None):
        super(SANModel, self).__init__()
        self.mode = mode
        self.features_dir = features_dir
        
        #set image_channel for image embedding , with output_size = emb_size=2048 #
        self.image_channel = ImageEmbedding(output_size=emb_size, mode=mode, extract_img_features=extract_img_features,
                                            features_dir=features_dir)

        # question embedding from 500 to 2048
        self.word_emb_size = word_emb_size
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)
        self.ques_channel = QuesEmbedding(
            word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        
        # attention: 2048-->512 #
        self.san = nn.ModuleList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_size, output_size))

    def forward(self, images, questions, image_ids):
        
        # use pretrained cnn to do image embedding#
        image_embeddings = self.image_channel(images, image_ids)
        
        #use LSTM to do word embedding #
        embeds = self.word_embeddings(questions)

        # #
        ques_embeddings = self.ques_channel(embeds)
        vi = image_embeddings
        u = ques_embeddings
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u)
        return output


# In[2]:


#import torchvision.models as models
#res152 = models.resnet152(pretrained=True)


# In[7]:


#vgg16 = models.vgg16(pretrained=True)


# In[21]:


#res152


# In[23]:


#nn.Sequential(res152.conv1,res152.bn1,res152.relu).parameters


# In[ ]:




