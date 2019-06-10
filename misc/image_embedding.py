import torch
import torch.nn as nn
from torch.autograd import Variable


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, use_gpu, feature_type='VGG'):
        super(ImageEmbedding, self).__init__()

        if feature_type == 'VGG':
            self.img_features = 512
        elif feature_type == 'Residual':
            self.img_features = 2048
        else:
            print('Unsupported feature type: \'{}\''.format(feature_type))

        self.hidden_size = hidden_size
        # linear: 512 -> 1024 #
        self.linear = nn.Linear(self.img_features, self.hidden_size)
        self.tanh = nn.Tanh()
        self.Dropout = nn.Dropout(0.5)
        self.use_gpu = use_gpu

    def forward(self, input_image):
        batch_size = input_image.shape[0]

        # batch_size * 14 * 14 * 512 -> batch_size * 196 * 512
        input_image = input_image.view(batch_size, -1, self.img_features)

        # batch_size * 196* 512 -> batch_size * 196 * 1024
        input_image = self.tanh(self.linear(input_image))

        return self.Dropout(input_image)
