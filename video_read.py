# Call like:
# python video_read.py -f data/videos/vid.mp4 -o data/faces -p faces_name -m 100000 -r 10000

from __future__ import print_function, division
import random
import time
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_video, write_video
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="Path to video file. Like: 'data/videos/video.mp4'")
parser.add_argument("-o", "--output_dir", help="Path to images out dir. Like: 'output/dir1'")
parser.add_argument("-p", "--output_filename_prefix", help="Prefix. Like: 'name'")
parser.add_argument("-m", "--max_pts", help="Max number of pts. Example: '100000'")
parser.add_argument("-r", "--pts_per_round", help="Number of pts per round. Example: '10000'")
args = parser.parse_args()
filename = args.filename
print('filename=', filename)
output_dir = args.output_dir
print('output=', output_dir)
output_filename_prefix = args.output_filename_prefix
print('output_filename_prefix=', output_filename_prefix)
max_pts = int(args.max_pts)
print('max_pts=', max_pts)
# Too many pts per round can cause OOM.
pts_per_round = int(args.pts_per_round)
print('pts_per_round=', pts_per_round)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # so transpose as
        image = image.transpose((2, 0, 1))
        # and btw inverse would be = image.transpose((1, 2, 0))
        #####
        return {'image': torch.from_numpy(image)}

trsf=transforms.Compose([
    Rescale(1000),
    ToTensor()
])

pts_processed = 0
end_of_file = False
name_pad = 1000000
r = 0

while pts_processed < max_pts and not end_of_file:
    vframes, aframes, meta = read_video(filename, start_pts=pts_processed, end_pts=(pts_processed + pts_per_round))
    print('vframes.len = %d' % len(vframes))
    print('aframes.len = %d' % len(aframes))
    print(meta)
    print('max_pts = %d, pts_processed = %d' % (max_pts, pts_processed))
    r += 1
    pts_processed += pts_per_round
    end_of_file = (len(vframes) == 1)
    # Now write the img files from the frames read.
    for i in range(len(vframes)):
        sample = trsf(vframes[i])
        output_filename = '%s/%s_%d_%d.jpg' % (output_dir, output_filename_prefix, r, (name_pad + i))
        print('write %s' % output_filename)
        utils.save_image(sample['image'], output_filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
