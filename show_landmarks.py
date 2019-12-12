# Call like:
# python show_landmarks.py -i data/faces/subdir -s 0 -n 10

from __future__ import print_function, division
import sys
import random
import time
import os
import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import my_lib
from my_lib import MyUtils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Path to data dir. Like: 'data/faces/subdir'")
parser.add_argument("-s", "--start", help="Index of first image to show. Like: 0")
parser.add_argument("-n", "--number", help="Number of faces to show. Like: 10")
args = parser.parse_args()
dir = args.input_dir
print('dir=', dir)
start = int(args.start)
print('start_index=', start)
number = int(args.number)
print('number=', number)

print('start')

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

transformed_dataset = my_lib.FaceLandmarksDataset(csv_file='%s/face_landmarks.csv' % dir,
                                                  root_dir='%s/' % dir,
                                                  transform=transforms.Compose([
                                                    my_lib.Rescale(1000),
                                                    my_lib.ToTensor()
                                                  ]))

count = 0
print('Iterate over dataset and apply.')
for i in range(start, min(number, len(transformed_dataset))):
    count += 1
    print('count = %d', count)
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())

    fig = plt.figure()
    image_print_format = sample['image'].numpy().transpose(1, 2, 0)
    MyUtils.show_landmarks(image_print_format, sample['landmarks'])
    plt.show()
