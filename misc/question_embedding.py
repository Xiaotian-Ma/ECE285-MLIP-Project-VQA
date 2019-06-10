import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, use_gpu, dict_json_path, bidirectional_or_not=False,
                 num_layers=1, dropout=0.5):
        super(QuestionEmbedding, self).__init__()
        self.use_gpu = use_gpu
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional_or_not
        if self.bidirectional == False:
            self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout,
                                bidirectional=self.bidirectional)
            self.num_directions = 1
        else:
            self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=2 * hidden_size, num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout,
                                bidirectional=self.bidirectional)
            self.num_directions = 2

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dict_json_path = dict_json_path

    def forward(self, ques_vec, method):
        if method == 1:
            # B: batch_size
            # W: question dimension
            B, W = ques_vec.size()
            # B * W -> B * W * (vocab_size+1)
            one_hot_vec = torch.zeros(B, W, self.vocab_size + 1).scatter_(2, ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)
            # To remove additional column in one_hot, use slicing
            one_hot_vec = Variable(one_hot_vec[:, :, 1:], requires_grad=False)
            if self.use_gpu and torch.cuda.is_available():
                one_hot_vec = one_hot_vec.cuda()
            x = self.lookuptable(one_hot_vec)
            emb_vec = self.tanh(x)

        if method == 2:
            B, W = ques_vec.size()
            self.word_dict = json.load(open(self.dict_json_path))['word_to_ix']
            self.word_vocab_size = len(self.word_dict)
            self.word_emb = nn.Embedding(self.vocab_size, self.emb_size)

            if self.use_gpu and torch.cuda.is_available():
                self.word_emb = self.word_emb.cuda()

            ques_vec = ques_vec.long()
            sentence_index = Variable(ques_vec, requires_grad=False)
            sentence_embed = self.word_emb(sentence_index)
            emb_vec = self.tanh(sentence_embed)

        final_output = torch.zeros(B, self.hidden_size)
        final_output = final_output.cuda()
        for i in range(emb_vec.shape[0]):
            h_0 = torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size)
            c_0 = torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size)
            h_0, c_0 = h_0.cuda(), c_0.cuda()
            for j in range(emb_vec.shape[1]):
                x_t = emb_vec[i][j].unsqueeze(0)
                x_t = x_t.unsqueeze(0)
                output_t, (h_t, c_t) = self.LSTM(x_t, (h_0, c_0))
                h_0 = h_t
                c_0 = c_t
            final_output[i] = h_t.squeeze(0).squeeze(0)

        return final_output

