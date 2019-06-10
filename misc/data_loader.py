import h5py
import torch
import numpy as np
from misc.utils import *
import json


class CDATA(torch.utils.data.Dataset):  # Extend PyTorch's Dataset class
    def __init__(self, opt, train, transform=None, quiet=False):
        if not quiet:
            print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])

        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        if train == 1:
            if not quiet:
                print('DataLoader loading h5 image train file: ' + opt['img_feature_path'])
            self.img_feature_path = opt['img_feature_path']
            self.ques = h5_file['/ques_train'][:]
            self.ans = h5_file['/ans_train'][:]
            self.split = h5_file['/split_train'][:]
        else:
            if not quiet:
                print('DataLoader loading h5 image test file: ' + opt['img_feature_path'])

            self.img_feature_path = opt['img_feature_path']
            self.ques = h5_file['/ques_val'][:]
            self.ans = h5_file['/ans_val'][:]
            self.split = h5_file['/split_val'][:]

            txt_json = json.load(open(opt['txt_json_path']))
            self.raw_ques = txt_json['raw_ques_val'][:]
            self.raw_mc_ans = txt_json['raw_mc_ans_val'][:]
            self.raw_ans = txt_json['raw_ans_val'][:]

        h5_file.close()

        self.feature_type = opt['feature_type']
        self.train = train
        self.transform = transform

        if not quiet:
            print('DataLoader loading json file: %s' % opt['json_file'])

        json_file = json.load(open(opt['json_file'], 'r'))
        self.ix_to_word = json_file['ix_to_word']
        self.ix_to_ans = json_file['ix_to_ans']

        self.vocab_size = len(self.ix_to_word)
        self.seq_length = self.ques.shape[1]

    def __len__(self):
        return self.split.shape[0]

    def getVocabSize(self):
        return self.vocab_size

    def __getitem__(self, idx):
        if self.img_feature_path:
            if self.train == 1:
                if self.feature_type == 'VGG':
                    img = np.load(self.img_feature_path + '/image' + str(idx) + '.npy').transpose(0, 2, 3, 1)[0]
                elif self.feature_type == 'Residual':
                    img = np.load(self.img_feature_path + '/image' + str(idx) + '.npy').transpose(0, 2, 3, 1)[0]
                else:
                    print("Error(train): feature type error")
                question = np.array(self.ques[idx], dtype=np.int32)
                answer = np.array(self.ans[idx])
                if self.transform is not None:
                    img = self.transform(img)
                    question = self.transform(question)

                return (img, question, answer)

            else:
                if self.feature_type == 'VGG':
                    img = \
                    np.load('./data/image_feature_after_cnn/' + 'image' + str(idx) + '.npy').transpose(0, 2, 3, 1)[0]
                elif self.feature_type == 'Residual':
                    img = \
                    np.load('./data/image_feature_after_res152/' + 'image' + str(idx) + '.npy').transpose(0, 2, 3, 1)[0]
                else:
                    print("Error(val): feature type error")

                question = np.array(self.ques[idx], dtype=np.int32)
                answer = np.array(self.ans[idx])
                if self.transform is not None:
                    img = self.transform(img)
                    question = self.transform(question)

                raw_question = self.raw_ques[idx]
                raw_mc_answer = self.raw_mc_ans[idx]
                raw_answer = self.raw_ans[idx]

                return img, question, answer, raw_question, raw_mc_answer, raw_answer

        question = np.array(self.ques[idx], dtype=np.int32)
        answer = np.array(self.ans[idx])
        if self.transform is not None:
            img = self.transform(img)
            question = self.transform(question)

        return img, question, answer

    def getVocabSize(self):
        return self.vocab_size

    def getSeqLength(self):
        return self.seq_length

