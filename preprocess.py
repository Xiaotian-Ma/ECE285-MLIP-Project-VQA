#!/usr/bin/env python
# coding: utf-8

# ### Train on train and Test on val

# In[1]:


import os  
def file_name(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:  
            #if os.path.splitext(file)[1] == '.jpeg':  
            L.append(os.path.join(root, file))  
        return L 

train2014_img_list = file_name('/datasets/ee285f-public/VQA2017/train2014')
val2014_img_list = file_name('/datasets/ee285f-public/VQA2017/val2014')


# In[3]:


train2014_img_list


# In[2]:


import json
train = []
val = []
print('Loading annotations and questions...')
train_anno = json.load(open('/datasets/ee285f-public/VQA2017/v2_mscoco_train2014_annotations.json', 'r'))
val_anno = json.load(open('/datasets/ee285f-public/VQA2017/v2_mscoco_val2014_annotations.json', 'r'))

train_ques = json.load(open('/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
val_ques = json.load(open('/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_val2014_questions.json', 'r')) 
print("successfully loading 4 jsons")

print("start dealing with training images")
for i in range(len(train_anno['annotations'])):
    # COCO_train2014_000000186605.jpg  #
    image_path = "train2014/"+"COCO_train2014_"+"000000"+str(train_anno['annotations'][i]['image_id'])
    if '/datasets/ee285f-public/VQA2017/'+image_path+'.jpg' in train2014_img_list:
        if i<3:
            print("this is used to show that train image path exists")
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
    #mc_ans = train_anno['annotations'][i]['answers']
        mc_ans = list(set([w['answer'] for w in train_anno['annotations'][i]['answers']]))
        question = train_ques['questions'][i]['question']
        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

print("start dealing with val images")    
for i in range(len(val_anno['annotations'])):
    # val2014/COCO_val2014_000000324670 #
    image_path = "val2014/"+"COCO_val2014_"+"000000"+str(val_anno['annotations'][i]['image_id'])
    if '/datasets/ee285f-public/VQA2017/'+image_path+'.jpg' in val2014_img_list:
        if i<3:
            print("this is used to show that val image path exists")
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
    #mc_ans = val_anno['annotations'][i]['answers']
        mc_ans = list(set([w['answer'] for w in val_anno['annotations'][i]['answers']]))
        question = val_ques['questions'][i]['question']
        val.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
print("finished")

print('Training sample %d, Testing sample %d...' %(len(train), len(val)))
json.dump(train, open('/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_raw_train.json', 'w'))
json.dump(val, open('/datasets/home/home-03/98/898/wel031/san_vqa/data/vqa_raw_test.json', 'w'))
print("finished!")  


# In[ ]:





# In[2]:


'''import os
print("Saving models")
epoch = 100
checkpoint_path = '/datasets/home/home-03/98/898/wel031/san_vqa/train_model/'
model_dir = os.path.join(checkpoint_path , str(epoch))
os.mkdir(model_dir)'''


# In[ ]:




