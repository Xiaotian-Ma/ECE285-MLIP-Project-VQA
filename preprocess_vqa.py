#!/usr/bin/env python
# coding: utf-8

# ### Train on train and Test on val

# In[6]:


import json
train = []
val = []
print('Loading annotations and questions...')
train_anno = json.load(open('/datasets/ee285f-public/VQA2017/v2_mscoco_train2014_annotations.json', 'r'))
val_anno = json.load(open('/datasets/ee285f-public/VQA2017/v2_mscoco_val2014_annotations.json', 'r'))

train_ques = json.load(open('/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
val_ques = json.load(open('/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_val2014_questions.json', 'r')) 

for i in range(len(train_anno['annotations'])):
    ans = train_anno['annotations'][i]['multiple_choice_answer']
    question_id = train_anno['annotations'][i]['question_id']
    mc_ans = train_anno['annotations'][i]['answers']
    question = train_ques['questions'][i]['question']
    # COCO_train2014_000000186605.jpg  #
    image_path = "train2014/"+"coco_train_2014_"+"000000"+str(train_anno['annotations'][i]['image_id'])
    
    train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
for i in range(len(val_anno['annotations'])):
    ans = val_anno['annotations'][i]['multiple_choice_answer']
    question_id = val_anno['annotations'][i]['question_id']
    mc_ans = val_anno['annotations'][i]['answers']
    question = val_ques['questions'][i]['question']
    # val2014/COCO_val2014_000000324670 #
    image_path = "val2014/"+"coco_val_2014_"+"000000"+str(val_anno['annotations'][i]['image_id'])
    
    val.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
print("finished")

print('Training sample %d, Testing sample %d...' %(len(train), len(val)))
json.dump(train, open('vqa_raw_train.json', 'w'))
json.dump(test, open('vqa_raw_test.json', 'w'))
print("finished!")  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




