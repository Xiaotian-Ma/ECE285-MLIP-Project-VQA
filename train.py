import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from misc.data_loader import CDATA
from misc.image_embedding import ImageEmbedding
from misc.question_embedding import QuestionEmbedding
from misc.san import Attention
# from misc.mutan import MutanFusion
from misc.utils import *


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def adjust_learning_rate(optimizer, epoch, lr, learning_rate_decay_every):
    # Sets the learning rate to the initial LR decayed by 10 every learning_rate_decay_every epochs
    lr_tmp = lr * (0.5 ** (epoch // learning_rate_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp
    return lr_tmp


def main(params):
    torch.backends.cudnn.enabled = False
    # Construct Data loader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    print("constructing data loader")
    opt = {'feature_type': params['feature_type'], 'img_feature_path': params['input_img_train_path'],
           'h5_ques_file': params['input_ques_h5'], 'json_file': params['input_json'],
           'txt_json_path': params['txt_json_path']}
    train_dataset = CDATA(opt, train=True, quiet=(not params['print_params']))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=False)

    # Construct NN models
    print("Constructing NN models.............")

    question_model = QuestionEmbedding(vocab_size=train_dataset.getVocabSize(), emb_size=params['emb_size'],
                                       hidden_size=params['hidden_size'], use_gpu=params['use_gpu'],
                                       dict_json_path=params['input_json'], num_layers=params['rnn_layers'],
                                       dropout=params['dropout'])

    image_model = ImageEmbedding(hidden_size=params['hidden_size'], use_gpu=params['use_gpu'],
                                 feature_type=params['feature_type'])

    if params['combine_method'] == 'SAN':
        image_question_model = Attention(input_size=params['hidden_size'], att_size=params['att_size'],
                                         img_seq_size=params['img_seq_size'], output_size=params['output_size'],
                                         drop_ratio=params['dropout'])
    if params['combine_method'] == 'MUTAN':
        image_question_model = MutanFusion(input_dim=params['hidden_size'], out_dim=params['output_size'],
                                           num_layers=params['R'])

    print("putting three model on gpu")
    if params['use_gpu'] and torch.cuda.is_available():
        question_model.cuda()
        image_model.cuda()
        image_question_model.cuda()

    if params['resume_from_epoch'] > 1:

        load_model_dir = os.path.join(params['checkpoint_path'], str(params['resume_from_epoch'] - 1))
        print('Loading model files from folder: %s' % load_model_dir)
        question_model.load_state_dict(torch.load(os.path.join(load_model_dir, 'question_model.pkl')), strict=False)
        image_model.load_state_dict(torch.load(os.path.join(load_model_dir, 'image_model.pkl')), strict=False)
        image_question_model.load_state_dict(torch.load(os.path.join(load_model_dir, 'attention_model.pkl')),
                                             strict=False)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer_parameter_group = [{'params': question_model.parameters()}, {'params': image_model.parameters()},
                                 {'params': image_question_model.parameters()}]

    if params['optim'] == 'sgd':
        optimizer = torch.optim.SGD(optimizer_parameter_group, lr=params['learning_rate'], momentum=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(optimizer_parameter_group, lr=params['learning_rate'],
                                        alpha=params['optim_alpha'], eps=params['optim_epsilon'],
                                        momentum=params['momentum'])
    elif params['optim'] == 'adam':
        optimizer = torch.optim.Adam(optimizer_parameter_group, eps=params['optim_epsilon'], lr=params['learning_rate'])
    elif params['optim'] == 'rprop':
        optimizer = torch.optim.Rprop(optimizer_parameter_group, lr=params['learning_rate'])
    else:
        print('Unsupported optimizer: \'%s\'' % (params['optim']))
        return none

    # Start training

    all_loss_store = []
    loss_store = []
    lr_cur = params['learning_rate']

    min_loss = 0
    for epoch in range(params['resume_from_epoch'], params['epochs'] + 1):
        print("training at epoch", epoch)
        if epoch > params['learning_rate_decay_start']:
            lr_cur = adjust_learning_rate(optimizer, epoch - 1 - params['learning_rate_decay_start'] + params[
                'learning_rate_decay_every'], params['learning_rate'], params['learning_rate_decay_every'])

        print('Epoch: %d | lr: %f' % (epoch, lr_cur))
        running_loss = 0.0
        print("start training 2333333")
        for i, (image, question, ans) in enumerate(train_loader):
            image = Variable(image)
            question = Variable(question)
            ans = Variable(ans, requires_grad=False)
            if (params['use_gpu'] and torch.cuda.is_available()):
                image = image.cuda()
                question = question.cuda()
                ans = ans.cuda()

            optimizer.zero_grad()

            img_emb = image_model(image)
            ques_emb = question_model(question, params['method'])
            output = image_question_model(ques_emb, img_emb)
            ans = torch.tensor(ans, dtype=torch.long, device=device)
            try:
                loss = criterion(output, ans)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            all_loss_store += [loss.data.item()]
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()

            if not (i + 1) % params['losses_log_every']:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                    epoch, params['epochs'], i + 1,
                    train_dataset.__len__() // params['batch_size'], loss.data.item()))

            torch.cuda.empty_cache()

        print("Saving models")
        model_dir = os.path.join(params['checkpoint_path'], str(epoch))
        os.mkdir(model_dir)
        torch.save(question_model.state_dict(), os.path.join(model_dir, 'question_model.pkl'))
        torch.save(image_model.state_dict(), os.path.join(model_dir, 'image_model.pkl'))
        torch.save(image_question_model.state_dict(), os.path.join(model_dir, 'attention_model.pkl'))

        if epoch == 0:
            min_loss = running_loss / len(train_loader)
            best_question_model = question_model.state_dict()
            best_image_model = image_model.state_dict()
            best_image_question_model = image_question_model.state_dict()

        if epoch > 0:
            if running_loss / len(train_loader) < min_loss:
                min_loss = running_loss / len(train_loader)
                best_question_model = question_model.state_dict()
                best_image_model = image_model.state_dict()
                best_image_question_model = image_question_model.state_dict()

        loss_store.append(running_loss / len(train_loader))
        print("****************epoch ", epoch, "finished**********************************")

    print("*********************saving the best model************************************")
    model_dir = os.path.join(params['checkpoint_path'], 'best_model')
    os.mkdir(model_dir)
    torch.save(best_question_model, os.path.join(model_dir, 'best_question_model.pkl'))
    torch.save(best_image_model, os.path.join(model_dir, 'best_image_model.pkl'))
    torch.save(best_image_question_model, os.path.join(model_dir, 'best_image_question_model.pkl'))
    print("loss_store:", loss_store)

    print("Saving all losses to file")
    np.savetxt(os.path.join(params['checkpoint_path'], 'all_loss_store.txt'), np.array(all_loss_store), fmt='%f')
    print("saving loss for every epoch to file")
    np.savetxt(os.path.join(params['checkpoint_path'], 'loss_store.txt'), np.array(loss_store), fmt='%f')
    print(loss_store)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img_train_path', default='./data/image_feature_after_res152',
                        help='path to the npy containing the train image feature')
    parser.add_argument('--input_ques_h5', default='./data/vqa_data_prepro.hdf5',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_json', default='./data/vqa_ques_prepro.json', help='output json file')
    parser.add_argument('--txt_json_path', default='./data/vqa_txt_data_prepro.json',
                        help='json file stores the raw queston, mc_ans, ans file')
    parser.add_argument('--start_from', default='',
                        help='path to a model checkpoint to initialize model weights from. Empty = don\'t')
    parser.add_argument('--resume_from_epoch', default=1, type=int, help='load model from previous epoch')

    # Options
    parser.add_argument('--feature_type', default='Residual', help='VGG or Residual')
    parser.add_argument('--combine_method', default='SAN', help='Mutan/SAN')
    parser.add_argument('--R', default=5, type=int, help='loop size for mutan')
    parser.add_argument('--emb_size', default=500, type=int, help='the size after embeeding from onehot')
    parser.add_argument('--hidden_size', default=1024, type=int, help='the hidden layer size of the model')
    parser.add_argument('--rnn_size', default=1024, type=int,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--att_size', default=512, type=int, help='size of sttention vector which refer to k in paper')
    parser.add_argument('--batch_size', default=200, type=int,
                        help='what is the utils batch size in number of images per batch? '
                             '(there will be x seq_per_img sentences)')
    parser.add_argument('--output_size', default=101, type=int, help='number of output answers for single answer')
    parser.add_argument('--rnn_layers', default=1, type=int, help='number of the rnn layer')
    parser.add_argument('--img_seq_size', default=196, type=int, help='number of feature regions in image')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio in network')
    parser.add_argument('--epochs', default=8, type=int, help='Number of epochs to run')
    parser.add_argument('--method', default=2, type=int, help='Ways to do question embedding')

    # Optimization
    parser.add_argument('--optim', default='rmsprop', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', default=6e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--learning_rate_decay_start', default=10, type=int,
                        help='at what epoch to start decaying learning rate?')
    parser.add_argument('--learning_rate_decay_every', default=10, type=int,
                        help='every how many epoch thereafter to drop LR by 0.1?')
    parser.add_argument('--optim_alpha', default=0.99, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--optim_beta', default=0.995, type=float, help='beta used for adam')
    parser.add_argument('--optim_epsilon', default=1e-8, type=float,
                        help='epsilon that goes into denominator in rmsprop')
    parser.add_argument('--max_iters', default=-1, type=int,
                        help='max number of iterations to run for (-1 = run forever)')
    parser.add_argument('--iterPerEpoch', default=1250, type=int, help=' no. of iterations per epoch')

    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_every', default=500, type=int, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', default='./train_model/update_res_san_101_train7/',
                        help='folder to save checkpoints into (empty = this folder)')

    # Visualization
    parser.add_argument('--losses_log_every', default=10, type=int,
                        help='How often do we save losses, for inclusion in the progress dump? (0 = disable)')

    # misc
    parser.add_argument('--use_gpu', default=1, type=int, help='to use gpu or not to use, that is the question')
    parser.add_argument('--id', default='1',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--gpuid', default=2, type=int, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', default=1234, type=int, help='random number generator seed to use')
    parser.add_argument('--print_params', default=1, type=int, help='pass 0 to turn off printing input parameters')

    args = parser.parse_args()
    params = vars(args)

    if params['print_params']:
        print('parsed input parameters:')
        print(json.dumps(params, indent=2))

    main(params)
