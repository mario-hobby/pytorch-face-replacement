from __future__ import print_function, division
import sys
import random
import time
import os
import torch
import argparse
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from my_lib import FaceLandmarksDataset, Rescale, RandomCrop, FaceCrop, ToTensor, KernelCrop
from my_lib import MyUtils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Path to data dir. Like: 'data/faces/subdir'")
parser.add_argument("-o", "--output_dir", help="Path to data dir. Like: 'output/subdir'")
parser.add_argument("-n", "--number", help="Number of faces to show. Like: 10")
# read arguments from the command line
args = parser.parse_args()
dir = args.input_dir
print('dir=', dir)
output_dir = args.output_dir
print('output_dir=', output_dir)
number = int(args.number)
print('number=', number)

print('start')

transformed_dataset = FaceLandmarksDataset(csv_file='%s/face_landmarks.csv' % dir,
                                           root_dir='%s/' % dir,
                                           transform=transforms.Compose([
                                               Rescale(1024),
                                               #RandomCrop(100),
                                               FaceCrop(),#KernelCrop(),
                                               Rescale(128),
                                               ToTensor()
                                           ]))

# Face and brows [0:27]
#   jaw-cheek line: [0:17]  # 17
#   brows [17:27]           # 10
# Eyes, nose, mouth [27:67]
#   nose [27:36]            # 9
#   eyes [36:48]            # 12
#   mouth [48:68]           # 20

start = time.time()
print('Iterate over dataset and apply.')
for i in range(0, min(number, len(transformed_dataset))):
    print('count = %d', i)
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())
    # resize once, just once and then erase this line
    landmarks = sample['landmarks'][26:27]# eye tear glands.
    print('new landmarks size')
    print(landmarks.size())

    fig = plt.figure()
    image_print_format = sample['image'].numpy().transpose(1, 2, 0)
    MyUtils.show_landmarks(image_print_format, landmarks)
    plt.show()

    filename = '%s/crop_%d.jpg' % (output_dir, i)
    utils.save_image(sample['image'], filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    print(sample['image'][0][0][1])

end = time.time()
