#!/usr/bin/env python
# coding: utf-8

# ### Train on train and Test on val

# In[10]:


import json
import os 

# download the VQA questions # 
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/')
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/')
  
  
  # Download the VQA Annotations :
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/')
os.system('wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/')
  
  # Let us now unzip the annotations :
os.system('unzip zip/v2_Questions_Train_mscoco.zip -d questions_and_annotations/')
os.system('unzip zip/v2_Questions_Val_mscoco.zip -d questions_and_annotations/')
os.system('unzip zip/v2_Annotations_Train_mscoco.zip -d questions_and_annotations/')
os.system('unzip zip/v2_Annotations_Val_mscoco.zip -d questions_and_annotations/')
  


# In[34]:


train = []
test = []

print('Loading annotations and questions...')
train_anno = json.load(open('../san_vqa/questions_and_annotations/v2_mscoco_train2014_annotations.json', 'r'))
val_anno = json.load(open('../san_vqa/questions_and_annotations/v2_mscoco_val2014_annotations.json', 'r'))

train_ques = json.load(open('../san_vqa/questions_and_annotations/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
val_ques = json.load(open('../san_vqa/questions_and_annotations/v2_OpenEnded_mscoco_val2014_questions.json', 'r')) 

for i in range(len(train_anno['annotations'])):
    ans = train_anno['annotations'][i]['multiple_choice_answer']
    question_id = train_anno['annotations'][i]['question_id']
    mc_ans = train_anno['annotations'][i]['answers']
    question = train_ques['questions'][i]['question']
    
    # train2014/COCO_train2014_000000011304.jpg #
    image_path = "train2014"+"coco_train_2014_"+"0000000"+str(train_anno['annotations'][i]['image_id'])

    
    
    
    train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
for i in range(len(val_anno['annotations'])):
    ans = val_anno['annotations'][i]['multiple_choice_answer']
    question_id = val_anno['annotations'][i]['question_id']
    mc_ans = val_anno['annotations'][i]['answers']
    question = val_ques['questions'][i]['question']
        
    # val2014/COCO_val2014_000000324670 #
    image_path = "val2014"+"coco_val_2014_"+"000000"+str(val_anno['annotations'][i]['image_id'])
    
   
    test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
print("finished!")    


# In[35]:


print('Training sample %d, Testing sample %d...' %(len(train), len(test)))
json.dump(train, open('vqa_raw_train.json', 'w'))
json.dump(test, open('vqa_raw_test.json', 'w'))


# In[31]:


'''eg  = [train_anno2['questions'][0],train_anno2['questions'][1]]
json.dump(eg, open('eg.json', 'w'))'''


# In[33]:


#train_anno2['questions'][0]


# In[17]:


#train_anno['annotations'][0]


# In[ ]:




