import os
import json

def file_name(file_dir):
    L=[]   
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))  
        return L 

train2014_img_list = file_name('/datasets/ee285f-public/VQA2017/train2014')
val2014_img_list = file_name('/datasets/ee285f-public/VQA2017/val2014')

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
    image_path = "train2014/"+"COCO_train2014_"+"000000"+str(train_anno['annotations'][i]['image_id'])
    if '/datasets/ee285f-public/VQA2017/'+image_path+'.jpg' in train2014_img_list:
        if i<3:
            print("this is used to show that train image path exists")
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        mc_ans = list(set([w['answer'] for w in train_anno['annotations'][i]['answers']]))
        question = train_ques['questions'][i]['question']
        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})

print("start dealing with val images")    
for i in range(len(val_anno['annotations'])):
    image_path = "val2014/"+"COCO_val2014_"+"000000"+str(val_anno['annotations'][i]['image_id'])
    if '/datasets/ee285f-public/VQA2017/'+image_path+'.jpg' in val2014_img_list:
        if i<3:
            print("this is used to show that val image path exists")
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        mc_ans = list(set([w['answer'] for w in val_anno['annotations'][i]['answers']]))
        question = val_ques['questions'][i]['question']
        val.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
print("finished")

print('Training sample %d, Testing sample %d...' %(len(train), len(val)))
json.dump(train, open('./data/vqa_raw_train.json', 'w'))
json.dump(val, open('./data/vqa_raw_test.json', 'w'))
print("finished!")  





