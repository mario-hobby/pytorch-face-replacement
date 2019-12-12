# Call like:
# python replace_faces.py -s data/faces/person_a/ -t data/faces/person_b/ -o output/dir/ -f result

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
from face_hasher import FaceHasher as face_hasher

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source_dir", help="Path to data dir. Like: 'data/faces/subdir'")
parser.add_argument("-t", "--target_dir", help="Path to data dir. Like: 'data/faces/subdir'")
parser.add_argument("-o", "--output_dir", help="Path to data dir. Like: 'output/subdir/'")
parser.add_argument("-f", "--output_filename", help="Name of the file to write. to data dir. Like: 'videoname'")
args = parser.parse_args()
source_dir = args.source_dir
target_dir = args.target_dir
output_dir = args.output_dir
output_filename = args.output_filename

print('start')
filename_index_pad = 100000

def get_hashable_feature(landmarks, scale):
    scaled_feature = landmarks * torch.tensor([0.95, 0.95]).double()
    return scaled_feature

max_collisions_per_hash=100
resolution_min_pow = 7
scale_pow = 8
resolution_max_pow = scale_pow

resolutions = 3

source_dataset = my_lib.FaceLandmarksDataset(csv_file='%s/face_landmarks.csv' % source_dir,
                                             root_dir='%s/' % source_dir,
                                             transform=transforms.Compose([
                                                my_lib.Rescale(1000),# Do not scale higher than original for write_vid to work
                                             ]))
print(source_dataset[0]['image'].shape)

target_dataset = my_lib.FaceLandmarksDataset(csv_file='%s/face_landmarks.csv' % target_dir,
                                             root_dir='%s/' % target_dir,
                                             transform=transforms.Compose([
                                                my_lib.Rescale(1000)# Do not scale higher than original for write_vid to work
                                             ]))
print(target_dataset[0]['image'].shape)

replace_face = my_lib.ReplaceFace()
crop_face = my_lib.FaceCrop()
to_tensor = my_lib.ToTensor()
scaled_face = transforms.Compose([my_lib.FaceCrop(), my_lib.Rescale(128)])
scaled_mouth = transforms.Compose([my_lib.MouthCrop(), my_lib.Rescale(128), my_lib.Rescale(2**scale_pow), my_lib.ToTensor()])
scaled_kernel = transforms.Compose([my_lib.KernelCrop(), my_lib.Rescale(128), my_lib.Rescale(2**scale_pow), my_lib.ToTensor()])
compare_faces=my_lib.CompareLandmarks()
face_coordinates = my_lib.FaceCoordinates()
scaled_to_source = my_lib.RescaleFlexible()

# Face and brows [0:27]
#   jaw-cheek line: [0:17]  # 17
#   brows [17:27]           # 10
# Eyes, nose, mouth [27:67]
#   nose [27:36]            # 9
#   eyes [36:48]            # 12
#   mouth [48:68]           # 20

hashes = {}

collision_fill_count = 0
count = 0
print('Iterate over target dataset and make map.')
start = time.time()
for i in range(len(target_dataset)):
    count += 1
    print('count = %d', count)
    face = scaled_face(target_dataset[i])
    kernel = scaled_kernel(target_dataset[i])
    image, landmarks = kernel['image'], kernel['landmarks']
    feature = get_hashable_feature(landmarks, 2**scale_pow)

    for r in range(1, resolutions + 1):
        hash = face_hasher.hash(r, face['landmarks'])

        if hash in hashes:
            if len(hashes[hash]) < max_collisions_per_hash:
                hashes[hash] = hashes[hash] + [i]
            else:
                collision_fill_count += 1
        else:
            hashes[hash] = [i]
        print('hash: %s len=%d' % (hash, len(hashes[hash])))

end = time.time()

print('hashes')
print('DONE making map. Processed count = %d in %d seconds' % (count, (end - start)))
print('collision_fill_count %d' % collision_fill_count)
print('total hashed keys = %d' % len(hashes.keys()))

print('Iterate over source dataset and find matches.')

total_diff = 0
hash_miss_count = 0
last_hash = ''
count = 0
start = time.time()

for i in range(len(source_dataset)):
    print('source example count = %d' % count)
    count += 1
    face = scaled_face(source_dataset[i])
    kernel = scaled_kernel(source_dataset[i])
    image, landmarks = kernel['image'], kernel['landmarks']
    feature = get_hashable_feature(landmarks, 2**scale_pow)
    hash_hit = False
    r = 1

    while not hash_hit and r <= resolutions:
        print('resolution = %d' % r)
        hash = face_hasher.hash(r, face['landmarks'])
        print('hash = %s' % hash)
        if hash in hashes:
            hash_hit = True
            print('hash HIT %s with collisions = %d' % (hash, len(hashes[hash])))
            indexes = hashes[hash]
            min_diff = sys.maxint
            one_start = time.time()
            choice = 0

            for j in range(len(indexes)):
                diff = compare_faces(scaled_face(source_dataset[i]), scaled_face(target_dataset[indexes[j]]))
                one_end = time.time()
                if diff < min_diff:
                    min_diff = diff
                    choice = indexes[j]

            one_end = time.time()
            total_diff += min_diff
            print('min_diff: %d' % min_diff)
            print('running average %d' % (total_diff / count))
            print('choice %d' % choice)
            print('hash %s and last_hash %s' % (hash, last_hash))
            last_hash = hash
            print('---------- time to match: %d ---------- ' % (one_end - one_start))

            target_face = crop_face(target_dataset[choice])
            source_face_coor = face_coordinates(source_dataset[i])
            target_face = scaled_to_source(target_face, (source_face_coor['height'], source_face_coor['width']))

            replace_start = time.time()
            result = replace_face(source_dataset[i], target_face)
            replace_end = time.time()
            print('---------- time to replace: %d ---------- ' % (replace_end - replace_start))

            sample = to_tensor(result)

            filename = '%s/%s_%d.jpg' % (output_dir, output_filename, filename_index_pad+i)
            utils.save_image(sample['image'], filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        else:
            r += 1

    if not hash_hit:
        hash_miss_count += 1
        print('hash MISS: %s' % hash)

end = time.time()
print('DONE checking hashes. Processed count = %d in %d seconds, at %d seconds per example' % (count, (end - start), (end - start)/count))
print('%d examples/second' % (count/(end - start)))
print('ave diff = %d' % (total_diff/count))
print('hash_misses=%d out of count=%d in source=%d' % (hash_miss_count, count, len(target_dataset)))
