#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from misc.data_loader import CDATA
from misc.Image_embedding_new import ImageEmbedding
from misc.question_embedding import QuestionEmbedding
from misc.san import Attention
import misc.utils 
import utils
from utils import *


# In[2]:


def main(params):
    print("constructing test dataset")
    opt ={ 'feature_type': params['feature_type'],'img_feature_path' : params['input_img_test_path'], 'h5_ques_file': params['input_ques_h5'],'json_file': params['input_json'] ,'txt_json_path': params['txt_json_path']}
    test_dataset = CDATA(opt, train=False, quiet=( not params['print_params']))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['batch_size'],shuffle=False)
    
    print("constructing nn models")
    vocab_size = test_dataset.getVocabSize()
    question_model = QuestionEmbedding(vocab_size = test_dataset.getVocabSize(), emb_size = params['emb_size'], hidden_size = params['hidden_size'], use_gpu = params['use_gpu'], num_layers = params['rnn_layers'], dropout = params['dropout'])
    image_model = ImageEmbedding(params['hidden_size'], params['feature_type'])
    attention_model = Attention(params['hidden_size'], params['att_size'],params['img_seq_size'], params['output_size'],params['dropout'])
    
    if params['use_gpu'] and torch.cuda.is_available():
        question_model.cuda()
        image_model.cuda()
        attention_model.cuda()   
        
    question_model.load_state_dict(torch.load(os.path.join(params['checkpoint_path'], 'question_model.pkl')),strict=False)
    image_model.load_state_dict(torch.load(os.path.join(params['checkpoint_path'], 'image_model.pkl')),strict=False)
    attention_model.load_state_dict(torch.load(os.path.join(params['checkpoint_path'], 'attention_model.pkl')),strict=False)
    
    correct, total = 0, 0
    for i, (image, question, ans, raw_question, raw_mc_answer, raw_answer) in enumerate(test_loader):
        image = Variable(image, requires_grad=False)
        question = Variable(question, requires_grad=False)
        if (params['use_gpu'] and torch.cuda.is_available()):
            image = image.cuda()
            question = question.cuda()
            
        img_emb = image_model(image)
        ques_emb = question_model(question)
        output = attention_model(ques_emb, img_emb)

        _, prediction = torch.max(output.data.item(), 1)
        total += ans.size(0)
        correct += (prediction.cpu() == ans).sum()
        if not (i+1)%100:
            print('Accuracy on %d images: %f%%'%(total, 100.0*correct/total))
            
    print('Accuracy on test set with %d images: %f %%'%(total, 100.0 * correct/total))


# In[3]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_img_test_path', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/val_image_feature_after_cnn', help='path to the h5file containing the train image feature')
    #parser.add_argument('--input_img_test_h5', default='data/vqa_data_img_vgg_test.h5', help='path to the h5file containing the test image feature')
    parser.add_argument('--input_ques_h5', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_data_prepro.hdf5', help='path to the json file containing additional info and vocab')

    parser.add_argument('--input_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_data_prepro.json', help='output json file')
    parser.add_argument('--txt_json_path', default = '/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_txt_data_prepro.json', help='json file stores the raw queston, mc_ans, ans file')
    parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = don\'t')

    # Options
    parser.add_argument('--feature_type', default='VGG', help='VGG or Residual')
    parser.add_argument('--emb_size', default=500, type=int, help='the size after embeeding from onehot')
    parser.add_argument('--hidden_size', default=1024, type=int, help='the hidden layer size of the model')
    parser.add_argument('--rnn_size', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--att_size', default=512, type=int, help='size of sttention vector which refer to k in paper')
    parser.add_argument('--batch_size', default=200, type=int, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--output_size', default=6, type=int, help='number of output answers for single answer')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of the rnn layer')
    parser.add_argument('--img_seq_size', default=196, type=int, help='number of feature regions in image')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio in network')

    # loading model files
    parser.add_argument('--checkpoint_path', default='train_model/20/', help='folder to save/load checkpoints to/from (empty = this folder)')

    # misc
    parser.add_argument('--use_gpu', default=1, type=int, help='to use gpu or not to use, that is the question')
    parser.add_argument('--id', default='1', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--gpuid', default=2, type=int, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', default=1234, type=int, help='random number generator seed to use')
    parser.add_argument('--print_params', default=1, type=int, help='pass 0 to turn off printing input parameters')

    args = parser.parse_args([])
    params = vars(args) # convert to ordinary dict
    if params['print_params']:
        print('parsed input parameters:')
        print(json.dumps(params, indent = 2))
        
    
    print("start evaluating")
    main(params)


# In[ ]:




