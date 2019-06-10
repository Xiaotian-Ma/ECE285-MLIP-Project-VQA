import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import json
import h5py
import sys
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

vgg16model = tv.models.vgg16_bn(pretrained=True)
res152model = tv.models.resnet152(pretrained=True)


class ImageDataset(td.Dataset):
    def __init__(self, opt):
        super(ImageDataset, self).__init__()

        self.mode = opt['mode']
        self.feature_type = opt['feature_type']
        self.image_size = opt['image_size']
        self.input_image_path = opt['input_image_path']

        self.img_dict = open(self.input_image_path, encoding='utf-8')
        self.img_dict = json.load(self.img_dict)

        if self.feature_type == 'VGG':
            print("use vgg16 to extract features")
            self.model = vgg16model.features
            self.model.cuda()
            self.model.eval()
        else:
            print("use resnet152 to extract features")
            self.model = nn.Sequential(*list(res152model.children())[:-2])
            self.model.cuda()
            self.model.eval()

        self.img_list = [w + '.jpg' for w in list(self.img_dict['img_path_' + self.mode].values())]

    def number_of_images(self):
        return len(self.self.img_list)

    def forward(self):
        transform = tv.transforms.Compose([tv.transforms.Resize((448, 448)), tv.transforms.ToTensor(),
                                           tv.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        if self.feature_type == 'VGG':
            print("using vgg16 to extract features.......................")
            for i in range(len(self.img_list)):
                img = Image.open('/datasets/ee285f-public/VQA2017/' + self.img_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1, 3, 448, 448)
                img = img.cuda()
                img = Variable(img)
                feature = self.model(img)
                # print("saving features at ",i,"th image")
                np.save("../data/" + self.mode + "_image_feature_after_vgg/" + "image" + str(i), feature.detach().cpu().float().numpy())

                if i < 10:
                    print("creating feature for ", i + 1, "out of 10 images")
                if i % 100 == 0:
                    sys.stdout.write(
                        "processing %d/%d (%.2f%% done)   \r" % (i, len(self.img_list), i * 100.0 / len(self.img_list)))
                    sys.stdout.flush()

        if self.feature_type == 'Residual':
            print("using residual network to extract features..........")
            for i in range(len(self.img_list)):
                img = Image.open('/datasets/ee285f-public/VQA2017/' + self.img_list[i]).convert('RGB')
                img = transform(img)
                img = img.view(1, 3, 448, 448)
                img = img.cuda()
                img = Variable(img)
                feature = self.model(img)
                np.save("../data/" + self.mode + "_image_feature_after_res152/" + "image" + str(i),
                        feature.detach().cpu().float().numpy())
                if i < 10:
                    print("creating feature for ", i + 1, "out of 10 images")
                    # break
                if i % 100 == 0:
                    sys.stdout.write(
                        "processing %d/%d (%.2f%% done)   \r" % (i, len(self.img_list), i * 100.0 / len(self.img_list)))
                    sys.stdout.flush()

        return 1


def main(params):
    image_dataset = ImageDataset(params)
    image_dataset.forward()
    print("well done!")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Options
    parser.add_argument('--feature_type', default='Residual', help='pretrained VGG or Residual network')
    parser.add_argument('--image_size', default=(448, 448), help='the given image size')
    parser.add_argument('--mode', default='train', help='train/val')
    parser.add_argument('--input_image_path', default='../data/image_path_output.json',
                        help='the json document storing the image path for train and val')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    main(params)
