#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import shuffle, seed
import sys
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import h5py
from nltk.tokenize import word_tokenize
import json
import re
import math

def get_image_path(img_json):
    return [w['img_path'].encode() for w in img_json]


def prepro_question(img_json, param_dict):
    #对所有的问题进行预处理——奏是分词啊
    for i,img in enumerate(img_json):
        s = img['question']
        if param_dict['token_method'] == 'nltk':
            # 用nltk包来分词
            txt = word_tokenize(str(s))
            img['processed_tokens'] = txt
        else:
            # 手动正则分词
            txt = [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", str(s)) if i!='' and i!=' ' and i!='\n']
            img['processed_tokens'] = txt
            
        #打出来看一些问题
        if i < 10: 
            print("showing first ",i+1,"out of 10 question")
            print(txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(img_json), i*100.0/len(img_json)) )
            sys.stdout.flush()  
            
    return img_json
        
            
def get_top_answers(img_json, param_dict):
    counts = {}
    for img in img_json:
        #提取答案#
        ans = img['ans']
        counts[ans] = counts.get(ans, 0)+1
    # 根据频率从高到低排列（越靠前的出现频率越大 #
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print(len(cw), "answers in total")
    vocab = []
    for i in range(param_dict['num_ans']):
        # vocab放的是出现频率高的答案 #
        vocab.append(cw[i][1])
    return vocab

def build_vocab_question(img_json, param_dict):
    count_thr = param_dict['word_count_threshold']
    counts = {}
    
    #统计所有问题里的单词的出现频数#
    for img in img_json:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w,0)+1
    
    #根据thr对所有的words进行筛选
    total_words = sum(counts.values())
    print("total words:", total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    
    vocab.append("UNK")
    for img in img_json:
        txt = img['processed_tokens']
        img['final_question'] = [w if counts.get(w,0)>count_thr else 'UNK' for w in txt]
    
    return img_json,vocab

def encode_question(img_json, param_dict, wtoi):
    # max_length: 截取部分的最大长度
    max_length = param_dict['max_length']
    N = len(img_json)
    
    #label_arrays: 存放最后所有question做完embedding之后的向量
    #label_length: 存放min（max_length, 问题长度）
    #question_id: 存放所有的question_id
    label_arrays = np.zeros((N,max_length), dtype = 'uint32')
    label_length = np.zeros(N, dtype = 'uint32')
    question_id = np.zeros(N,dtype = 'uint32')
    question_counter = 0
    
    for i,img in enumerate(img_json):
        question_id[i] = img['ques_id']
        label_length[i] = min(max_length, len(img['final_question']))
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
                
    return label_arrays, label_length, question_id
                
def apply_vocab_question(img_json, wtoi):
    for img in img_json:
        question = [w if w in wtoi else 'UNK' for w in img['processed_tokens']]
        img['final_question'] = question
    return img_json

def get_unqiue_img(img_json):
    count_img = {}
    N = len(img_json)
    img_pos = np.zeros(N, dtype='uint32')
    ques_pos_tmp = {}
    
    # 看每一个不同的img_path出现了几次 #
    for img in img_json:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1
    #直接提取 得到所有的unique image   
    unique_img = list(count_img.keys())
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} 
    
    for i, img in enumerate(img_json):
        # 得到每一个image_path对应的出现次数？#
        idx = imgtoi.get(img['img_path'])
        #img_pos存的是每个img_path的出现次数
        img_pos[i] = idx
        
        if idx-1 not in ques_pos_tmp:
            ques_pos_tmp[idx-1] = []
            
        ques_pos_tmp[idx-1].append(i+1)
        
    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N,3), dtype='uint32')
    ques_pos_len = np.zeros(img_N, dtype='uint32')
    
    for idx, ques_list in ques_pos_tmp.items():
        ques_pos_len[idx] = len(ques_list)
        for j in range(len(ques_list)):
            ques_pos[idx][j] = ques_list[j]
            
    return unique_img, img_pos, ques_pos, ques_pos_len

            
def encode_answer(img_json, atoi):
    N = len(img_json)
    ans_arrays = np.zeros(N, dtype='uint32')
    
    for i, img in enumerate(img_json):
        # 1000 意味着这个题目没有答案 #
        ans_arrays[i] = atoi.get(img['ans'], 1000) 
    return ans_arrays

'''def encode_mc_answer(img_json, atoi):
    N = len(imgs)
    # 我们把mc_answer的答案控制到了6个.....#
    mc_ans_arrays = np.zeros((N, 6), dtype='uint32')
    for i, img in enumerate(imgs):
        for j,ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans,0)
    return mc_ans_arrays

def answer_onehot_encoding(img_json):
    N = len(img_json)
    ans_arrays = np.zeros(N)
    for i,img in enumerate(img_json):
        #print(img['MC_ans'])
        #print(img['ans'])
        j = img['MC_ans'].index(img['ans'])
        ans_arrays[i] = j
    return ans_arrays'''
        
                
def filter_image_on_mc_answer(img_json, param_dict):
    #筛选出来长度大于等于3 小于等于6的图片
    return [w for w in img_json if len(w['MC_ans'])>=param_dict['minlength_of_mc_answer'] and len(w['MC_ans'])<=param_dict['maxlength_of_mc_answer']]
    

def get_raw_question(img_json):
    return [w['question'] for w in img_json]

def get_raw_mc_answer(img_json):
    return [w['MC_ans'] for w in img_json]

def get_raw_answer(img_json):
    return [w['ans'] for w in img_json]


import numpy as np
def main(params):
    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_val = json.load(open(params['input_val_json'], 'r'))
    

    
    #imgs_train = imgs_train[:30]
    #imgs_val = imgs_val[:30]
    
    #这样大概提取8w张train 6w多条val 基本是够了。。。。
    
    
    #  params['word2vector_method'] == 1: github里头的方法，用频率做word2vector
    if params['word2vector_method'] == 1:
        # 这里用的是做频率做answer和question embedding #
        top_answers = get_top_answers(imgs_train, params)
        atoi = {w:i for i,w in enumerate(top_answers)}
        itoa = {i:w for i,w in enumerate(top_answers)}
    
        # 选出来答案在top_answer里面的question（降维） #
        imgs_train = [img for img in imgs_train if img['ans'] in top_answers]
        
        # 选出来mulitiple_choice至少三个选项 至多六个选项的问题
        imgs_train = filter_image_on_mc_answer(imgs_train, params)
        imgs_val = filter_image_on_mc_answer(imgs_val, params)
        
        # 看看还剩了多少#
        print("number of training image left from previous 300000: ",len(imgs_train))
        print("number of training image left from previous 300000: ",len(imgs_val))
        
        # 提取id
        img_path_train = get_image_path(imgs_train)
        img_path_val = get_image_path(imgs_val)
    
        # 对问题进行分词 #
        imgs_train = prepro_question(imgs_train, params)
        imgs_val = prepro_question(imgs_val, params)
    
        # 对所有的question进行提取vocab #
        imgs_train, vocab = build_vocab_question(imgs_train, params)
        itow = {i+1:w for i,w in enumerate(vocab)}
        wtoi = {w:i+1 for i,w in enumerate(vocab)}
        
        #用vocab处理img_val#
        imgs_val = apply_vocab_question(imgs_val, wtoi)

        #用wtoi做question的embedding #
        ques_train, ques_len_train, ques_id_train = encode_question(imgs_train, params, wtoi)
        ques_val, ques_len_val, ques_id_val = encode_question(imgs_val, params, wtoi)
        
        # 提取文本问题 文本mc_answer，文本标准答案（为了后来做预测用）#
        raw_ques_train = get_raw_question(imgs_train)
        raw_ques_val = get_raw_question(imgs_val)
        
        raw_mc_ans_train = get_raw_mc_answer(imgs_train)
        raw_mc_ans_val = get_raw_mc_answer(imgs_val)
        
        raw_ans_train = get_raw_answer(imgs_train)
        raw_ans_val = get_raw_answer(imgs_val)

        # 对训练和测试集的答案做embedding
        ans_train = encode_answer(imgs_train,atoi)
        ans_val = encode_answer(imgs_val,atoi)
    
        # split
        N_train = len(imgs_train)
        N_val = len(imgs_val)
        
        # since the train image is already suffled, we just use the last val_num image as validation
        # train = 0, val = 1, test = 2
        split_train = np.zeros(N_train)
        split_val = np.zeros(N_val)
        split_val[:] = 2
        

        
        f = h5py.File(params['output_h5'], "w")
        

        f.create_dataset("img_path_train", data = img_path_train)
        f.create_dataset("img_path_val",data = img_path_val)
        
        f.create_dataset("ans_train", dtype='int32', data=ans_train)
        f.create_dataset("ans_val", dtype='int32', data=ans_val)
        
        f.create_dataset("ques_train", dtype='int32', data=ques_train)
        f.create_dataset("ques_val", dtype='int32', data=ques_val)
        
        f.create_dataset("ques_len_train", dtype='int32', data=ques_len_train)
        f.create_dataset("ques_len_val", dtype='int32', data=ques_len_val)
        
        f.create_dataset("ques_id_train", dtype='int32', data=ques_id_train)
        f.create_dataset("ques_id_val", dtype='int32', data=ques_id_val)
        
        f.create_dataset("split_train", dtype='int32', data=split_train)
        f.create_dataset("split_val", dtype='int32', data=split_val)
               
        f.close()
        print('wrote ', params['output_h5'])
        
        #output_dir ='/datasets/home/home-03/98/898/wel031/san_vqa/data/updated_vqa_prepro.json'
        #json.dump(f, open(output_dir , 'w'))
        #print("wrote", output_dir)
        
        # create output json file
        out = {}
        itoa[1000] = 'UNK'
        out['ix_to_word'] = itow
        out['ix_to_ans'] = itoa
        out['word_to_ix'] = wtoi
        out['ans_to_ix'] = atoi
        
        json.dump(out, open(params['output_json'], 'w'))
        print('wrote ', params['output_json'])
        
        raw_txt = {}
        raw_txt['raw_ques_train'] = raw_ques_train
        raw_txt['raw_ques_val'] = raw_ques_val
        
        raw_txt['raw_mc_ans_train'] = raw_mc_ans_train
        raw_txt['raw_mc_ans_val'] = raw_mc_ans_val

        raw_txt['raw_ans_train'] = raw_ans_train
        raw_txt['raw_ans_val'] = raw_ans_val
        
        json.dump(raw_txt, open(params['output_txt_json'], 'w'))
        print('wrote ', params['output_txt_json'])
        
    else:
        
        print("try a new way to do embedding")
        top_answers = get_top_answers(imgs_train, params)
        atoi = {w:i for i,w in enumerate(top_answers)}
        itoa = {i:w for i,w in enumerate(top_answers)}
        
        # 选出来答案在top_answer里面的question（降维） #
        imgs_train = [img for img in imgs_train if img['ans'] in top_answers]
        
        # 选出来mulitiple_choice至少三个选项 至多六个选项的问题
        imgs_train = filter_image_on_mc_answer(imgs_train, params)
        imgs_val = filter_image_on_mc_answer(imgs_val, params)
        
        # 看看还剩了多少#
        print("number of training image left from previous 300000: ",len(imgs_train))
        print("number of training image left from previous 300000: ",len(imgs_val))
        
        # 提取id
        img_path_train = get_image_path(imgs_train)
        img_path_val = get_image_path(imgs_val)
    
        # 对问题进行分词 #
        imgs_train = prepro_question(imgs_train, params)
        imgs_val = prepro_question(imgs_val, params)
        
        imgs_train, vocab = build_vocab_question(imgs_train, params)
        itow = {i+1:w for i,w in enumerate(vocab)}
        wtoi = {w:i+1 for i,w in enumerate(vocab)}
        
        #用vocab处理img_val#
        imgs_val = apply_vocab_question(imgs_val, wtoi)
        
        #用wtoi做question的embedding #
        ques_train = [img['processed_tokens'] for img in imgs_train]
        ques_val = [img['processed_tokens'] for img in imgs_val]
        
        # 提取文本问题 文本mc_answer，文本标准答案（为了后来做预测用）#
        raw_ques_train = get_raw_question(imgs_train)
        raw_ques_val = get_raw_question(imgs_val)
        
        raw_mc_ans_train = get_raw_mc_answer(imgs_train)
        raw_mc_ans_val = get_raw_mc_answer(imgs_val)
        
        raw_ans_train = get_raw_answer(imgs_train)
        raw_ans_val = get_raw_answer(imgs_val)
        
        ans_train = encode_answer(imgs_train,atoi)
        ans_val = encode_answer(imgs_val,atoi)
        
        print("saving numerical output")
        f = h5py.File(params['output2_h5'], "w")
        f.create_dataset("img_path_train", data = img_path_train)
        f.create_dataset("img_path_val",data = img_path_val)
        
        f.create_dataset("ans_train", dtype='int32', data=ans_train)
        f.create_dataset("ans_val", dtype='int32', data=ans_val)
        
        #f.create_dataset("split_train", dtype='int32', data=split_train)
        #f.create_dataset("split_val", dtype='int32', data=split_val)     
        f.close()
        print('wrote ', params['output2_h5'])
        
        print("saving train question and val question")
        ques_json = {}
        ques_json['ques_train'] = ques_train
        ques_json['ques_val'] = ques_val
        ques_json['ix_to_word'] = itow
        ques_json['ix_to_ans'] = itoa
        ques_json['word_to_ix'] = wtoi
        ques_json['ans_to_ix'] = atoi
        json.dump(ques_json, open(params['ques_output_json'],'w'))
        print("wrote ",params['ques_output_json'])
        
        '''print("saving txt file")
        raw_txt = {}
        raw_txt['raw_ques_train'] = raw_ques_train
        raw_txt['raw_ques_val'] = raw_ques_val
        raw_txt['raw_mc_ans_train'] = raw_mc_ans_train
        raw_txt['raw_mc_ans_val'] = raw_mc_ans_val
        raw_txt['raw_ans_train'] = raw_ans_train
        raw_txt['raw_ans_val'] = raw_ans_val
        json.dump(raw_txt, open(params['output_txt_json'], 'w'))
        print('wrote ', params['output_txt_json'])'''
    


# In[15]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_raw_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_val_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_raw_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--output_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_data_prepro.json', help='output json file')
    parser.add_argument('--output_txt_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_txt_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_data_prepro.hdf5', help='output h5 file')
    parser.add_argument('--output2_h5', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_data_prepro_method2.hdf5', help='output h5 file')
    parser.add_argument('--ques_output_json', default='/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_ques_prepro.json', help='output h5 file')
    
    
  
    # options
    # 这里有个区别，稍微把question的长度控制的更大一些……
    parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    parser.add_argument('--word2vector_method', default = 2, help = 'method to do word_embedding， 1 means probability, 2 means vector')
    parser.add_argument('--minlength_of_mc_answer', default =3, help = 'use this number to filter the data on mc answer')
    parser.add_argument('--maxlength_of_mc_answer', default =6, help = 'use this number to filter the data on mc answer')

    args = parser.parse_args([])
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    
    main(params)


# In[ ]:




