# Call like:
# python write_video.py -s input_dir -o output_dir -f video_name

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
from geo_spatial import GeoSpatial as spatial
from torchvision.io import read_video, write_video

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", help="Path to data dir. Like: 'data/faces/subdir'")
parser.add_argument("-o", "--output_dir", help="Path to data dir. Like: 'output/subdir/'")
parser.add_argument("-f", "--ouput_filename", help="Name of the file to write. to data dir. Like: 'videoname'")
args = parser.parse_args()
source_dir = args.source_dir
output_dir = args.output_dir
output_filename = args.ouput_filename

print('start')

max_collisions_per_hash=100
resolution_min_pow = 1
scale_pow = 8#8
resolution_max_pow = scale_pow

source_dataset = my_lib.FaceLandmarksDataset(csv_file='%s/face_landmarks.csv' % source_dir,
                                             root_dir='%s/' % source_dir)

one = torch.tensor([source_dataset[0]['image']])
images = torch.tensor([source_dataset[0]['image']])
print(images.shape)

last_choice = 0
last_diff = 0
total_diff = 0
hash_hit_count = 0
count = 0
start = time.time()
for i in range(1, len(source_dataset)):
    print('Source example count = %d' % count)
    count += 1
    t = torch.tensor(source_dataset[i]['image'])
    print(t.shape)
    images = torch.cat((images, one), 0)
    images[i] = t

end = time.time()
print('DONE checking hashes. Processed count = %d in %d seconds, at %d seconds per example' % (count, (end - start), (end - start)/len(source_dataset)))

print(images.shape)

filename = '%s/%s.mp4' % (output_dir, output_filename)
write_video(filename, images, np.int(30), video_codec='libx264', options=None)
