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
    for i, img in enumerate(img_json):
        s = img['question']
        if param_dict['token_method'] == 'nltk':
            txt = word_tokenize(str(s))
            img['processed_tokens'] = txt
        else:
            txt = [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", str(s)) if
                   i != '' and i != ' ' and i != '\n']
            img['processed_tokens'] = txt

        if i < 10:
            print("showing first ", i + 1, "out of 10 question")
            print(txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (i, len(img_json), i * 100.0 / len(img_json)))
            sys.stdout.flush()

    return img_json


def get_top_answers(img_json, param_dict):
    counts = {}
    for img in img_json:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print(len(cw), "answers in total")
    vocab = []
    for i in range(param_dict['num_ans']):
        vocab.append(cw[i][1])
    return vocab


def build_vocab_question(img_json, param_dict):
    count_thr = param_dict['word_count_threshold']
    counts = {}

    for img in img_json:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1

    total_words = sum(counts.values())
    print("total words:", total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]

    vocab.append("UNK")
    for img in img_json:
        txt = img['processed_tokens']
        img['final_question'] = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]

    return img_json, vocab


def encode_question(img_json, param_dict, wtoi):
    max_length = param_dict['max_length']
    N = len(img_json)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0

    for i, img in enumerate(img_json):
        question_id[i] = img['ques_id']
        label_length[i] = min(max_length, len(img['final_question']))
        for k, w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i, k] = wtoi[w]

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
    for img in img_json:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1
    unique_img = list(count_img.keys())
    imgtoi = {w: i + 1 for i, w in enumerate(unique_img)}

    for i, img in enumerate(img_json):
        idx = imgtoi.get(img['img_path'])
        img_pos[i] = idx

        if idx - 1 not in ques_pos_tmp:
            ques_pos_tmp[idx - 1] = []

        ques_pos_tmp[idx - 1].append(i + 1)

    img_N = len(ques_pos_tmp)
    ques_pos = np.zeros((img_N, 3), dtype='uint32')
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
        ans_arrays[i] = atoi.get(img['ans'], 1000)
    return ans_arrays


def answer_onehot_encoding(img_json):
    N = len(img_json)
    ans_arrays = np.zeros(N)
    for i, img in enumerate(img_json):
        j = img['MC_ans'].index(img['ans'])
        ans_arrays[i] = j
    return ans_arrays


def filter_image_on_mc_answer(img_json, param_dict):
    return [w for w in img_json if
            len(w['MC_ans']) >= param_dict['minlength_of_mc_answer']
            and len(w['MC_ans']) <= param_dict['maxlength_of_mc_answer']]


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

    if params['word2vector_method'] == 1:
        top_answers = get_top_answers(imgs_train, params)
        atoi = {w: i for i, w in enumerate(top_answers)}
        itoa = {i: w for i, w in enumerate(top_answers)}

        imgs_train = [img for img in imgs_train if img['ans'] in top_answers]

        imgs_train = filter_image_on_mc_answer(imgs_train, params)
        imgs_val = filter_image_on_mc_answer(imgs_val, params)

        print("number of training image left from previous 300000: ", len(imgs_train))
        print("number of training image left from previous 300000: ", len(imgs_val))

        img_path_train = get_image_path(imgs_train)
        img_path_val = get_image_path(imgs_val)

        imgs_train = prepro_question(imgs_train, params)
        imgs_val = prepro_question(imgs_val, params)

        imgs_train, vocab = build_vocab_question(imgs_train, params)
        itow = {i: w for i, w in enumerate(vocab)}
        wtoi = {w: i for i, w in enumerate(vocab)}

        imgs_val = apply_vocab_question(imgs_val, wtoi)

        ques_train, ques_len_train, ques_id_train = encode_question(imgs_train, params, wtoi)
        ques_val, ques_len_val, ques_id_val = encode_question(imgs_val, params, wtoi)

        raw_ques_train = get_raw_question(imgs_train)
        raw_ques_val = get_raw_question(imgs_val)

        raw_mc_ans_train = get_raw_mc_answer(imgs_train)
        raw_mc_ans_val = get_raw_mc_answer(imgs_val)

        raw_ans_train = get_raw_answer(imgs_train)
        raw_ans_val = get_raw_answer(imgs_val)

        ans_train = answer_onehot_encoding(imgs_train)
        ans_val = answer_onehot_encoding(imgs_val)

        # split
        N_train = len(imgs_train)
        N_val = len(imgs_val)

        # since the train image is already suffled, we just use the last val_num image as validation
        split_train = np.zeros(N_train)
        split_val = np.zeros(N_val)
        split_val[:] = 2

        f = h5py.File(params['output_h5'], "w")

        f.create_dataset("img_path_train", data=img_path_train)
        f.create_dataset("img_path_val", data=img_path_val)

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
        atoi = {w: i for i, w in enumerate(top_answers)}
        itoa = {i: w for i, w in enumerate(top_answers)}

        imgs_train = [img for img in imgs_train if img['ans'] in top_answers]

        imgs_train = filter_image_on_mc_answer(imgs_train, params)
        imgs_val = filter_image_on_mc_answer(imgs_val, params)

        print("number of training image left from previous 300000: ", len(imgs_train))
        print("number of training image left from previous 300000: ", len(imgs_val))

        img_path_train = get_image_path(imgs_train)
        img_path_val = get_image_path(imgs_val)

        imgs_train = prepro_question(imgs_train, params)
        imgs_val = prepro_question(imgs_val, params)

        imgs_train, vocab = build_vocab_question(imgs_train, params)
        itow = {i: w for i, w in enumerate(vocab)}
        wtoi = {w: i for i, w in enumerate(vocab)}

        imgs_val = apply_vocab_question(imgs_val, wtoi)

        ques_train = [img['processed_tokens'] for img in imgs_train]
        ques_val = [img['processed_tokens'] for img in imgs_val]

        raw_ques_train = get_raw_question(imgs_train)
        raw_ques_val = get_raw_question(imgs_val)

        raw_mc_ans_train = get_raw_mc_answer(imgs_train)
        raw_mc_ans_val = get_raw_mc_answer(imgs_val)

        raw_ans_train = get_raw_answer(imgs_train)
        raw_ans_val = get_raw_answer(imgs_val)

        ans_train = encode_answer(imgs_train, atoi)
        ans_val = encode_answer(imgs_val, atoi)

        print("saving numerical output")
        f = h5py.File(params['output2_h5'], "w")
        f.create_dataset("img_path_train", data=img_path_train)
        f.create_dataset("img_path_val", data=img_path_val)

        f.create_dataset("ans_train", dtype='int32', data=ans_train)
        f.create_dataset("ans_val", dtype='int32', data=ans_val)

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
        json.dump(ques_json, open(params['ques_output_json'], 'w'))
        print("wrote ", params['ques_output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='../data/vqa_raw_train.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_val_json', default='../data/vqa_raw_test.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int,
                        help='number of top answers for the final classifications.')

    parser.add_argument('--output_json', default='../data/vqa_data_prepro.json', help='output json file')
    parser.add_argument('--output_txt_json', default='../data/vqa_txt_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/vqa_data_prepro.hdf5', help='output h5 file')
    parser.add_argument('--output2_h5', default='../data/vqa_data_prepro_method2.hdf5', help='output h5 file')
    parser.add_argument('--ques_output_json', default='../data/vqa_ques_prepro.json', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=20, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    parser.add_argument('--word2vector_method', default=2, type=int,
                        help='method to do word_embedding， 1 means probability, 2 means vector')
    parser.add_argument('--minlength_of_mc_answer', default=3, type=int, help='use this number to filter the data on mc answer')
    parser.add_argument('--maxlength_of_mc_answer', default=6, type=int, help='use this number to filter the data on mc answer')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(params)
