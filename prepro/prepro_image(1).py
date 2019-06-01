#!/usr/bin/env python
# coding: utf-8

# In[6]:


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
import h5py
import sys
from torch.autograd import Variable

# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# ### 我现在想把prepro和image embedding写到一起……

vgg16model = tv.models.vgg16_bn(pretrained = True)
res152model = tv.models.resnet152(pretrained = True)

class ImageDataset(td.Dataset):
    def __init__(self, opt):
        super(ImageDataset, self).__init__()
        
        self.mode = opt['mode']
        self.feature_type = opt['feature_type']
        self.image_size = opt['image_size']
        self.input_image_path = opt['input_image_path']
        
        self.img_dict = open(self.input_image_path ,encoding='utf-8')
        self.img_dict = json.load(self.img_dict)
        
        if self.feature_type =='VGG':
            print("use vgg16 to extract features")
            self.model = vgg16model.features
            self.model.cuda()
            self.model.eval()
        else:
            print("use resnet to extract features")
            #awaiting to be completed
            #self.model = res152model.features
            
        # if mode == 'train': img_path_train
        # if mode == 'val': img_path_val
        self.img_list = [w+'.jpg' for w in list(self.img_dict['img_path_'+ self.mode].values())]
        
 
                             
    def number_of_images(self):
        return len(self.self.img_list)
    
    def forward(self):
        transform = tv.transforms.Compose([tv.transforms.Resize((448,448)),tv.transforms.ToTensor(),tv.transforms.Normalize(mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5))])
        
        cur_dir = '/datasets/home/home-03/98/898/wel031/san_vqa/data'
        folder_name = 'image_feature_after_cnn'
        if os.path.isdir(cur_dir):
            os.mkdir(os.path.join(cur_dir, folder_name))
          
        if self.feature_type =='VGG':
            
            for i in range(len(self.img_list)):
                img = Image.open('/datasets/ee285f-public/VQA2017/'+self.img_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1,3,448,448)
                img = img.cuda()
                img = Variable(img)
                feature = self.model(img)
                #print("saving features at ",i,"th image")
                np.save("/datasets/home/home-03/98/898/wel031/san_vqa/data/image_feature_after_cnn/"+"image"+str(i)    ,feature.detach().cpu().float().numpy())
                
                if i<10:
                    print("creating feature for ",i+1,"out of 10 images")
                   
                
                 
                if i % 100 == 0:
                    sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(self.img_list), i*100.0/len(self.img_list)) )
                    sys.stdout.flush() 
            
                
        if self.feature_type == 'Residual':
            print("using residual network to extract features..........")
            '''final_output = torch.ones(end_number-start_number+1 ,2048, 14, 14)
            for i in range(start_number,end_number):
                img = Image.open(self.val_path_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1,3,448,448)
                final_output[i] = self.model(img)'''
                
        return 1
                
            
        


# In[31]:


def main(params):
    image_dataset = ImageDataset(params)
    image_dataset.forward()
    #image_feature_after_network = image_dataset.forward()
    #if params['mode']=='train':
    #json_path = params['mode']+'_image_feature_output.json'
    #output_json = '/datasets/home/home-03/98/898/wel031/san_vqa/data/'+json_path
    #json.dump(image_feature_after_network , open(output_json, 'w'))
    print("well done!")


# In[32]:


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Options
    parser.add_argument('--feature_type', default='VGG', help='pretrained VGG or Residual network')
    parser.add_argument('--image_size', default=(448,448), help='the given image size')
    parser.add_argument('--mode', default='train', help='train/val')
    parser.add_argument('--input_image_path', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/image_path_output.json', help='the json document storing the image path for train and val')
    #parser.add_argument('--output_image_feature_path', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/'+'image_feature_output.json', help='the json document storing the image path for train and val')
    
    args = parser.parse_args([])
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    
    main(params)


# In[ ]:




