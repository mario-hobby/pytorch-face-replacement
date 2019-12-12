from __future__ import print_function, division
import sys
import random
import time
import math
import os
from scipy.spatial import ConvexHull
import torch
import pandas as pd
from skimage import io, transform
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MyUtils():
    @staticmethod
    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Helper function to show a batch
    @staticmethod
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                        landmarks_batch[i, :, 1].numpy() + grid_border_size,
                        s=10, marker='.', c='r')

            plt.title('Batch from dataloader')

class ImageDataset(Dataset):
    def __init__(self, root_dir, img_prefix, transform=None):
        self.root_dir = root_dir
        self.img_prefix = img_prefix
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                '%s%d.jpg' % (self.img_prefix, idx + 1000000))
        image = io.imread(img_name)
        sample = {'image': image, 'landmarks': None}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks, 'name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

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

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
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
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RescaleFlexible(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __call__(self, sample, output_size):
        assert isinstance(output_size, (int, tuple))
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(output_size, int):
            if h < w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class CompareLandmarks(object):
    def __call__(self, source, target):
        source_landmarks = source['landmarks']
        target_landmarks = target['landmarks']
        return mean_squared_error(source_landmarks, target_landmarks)

# Nose, Mouth, Eyes.
class CompareFaceCore(object):
    def __call__(self, source, target):
        source_landmarks = source['landmarks'][27:67]
        target_landmarks = target['landmarks'][27:67]
        return mean_squared_error(source_landmarks, target_landmarks)

class FaceCoordinates(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        left = -1
        right = 0
        top = -1
        bottom = 0
        for landmark in landmarks:
            if landmark[0] < left or left == -1:
                left = landmark[0]
            if landmark[1] < top or top == -1:
                top = landmark[1]
            if landmark[0] > right:
                right = landmark[0]
            if landmark[1] > bottom:
                bottom = landmark[1]

        return {'top': np.int(top),
                'left': np.int(left),
                'height': np.int(bottom - top),
                'width': np.int(right - left)}

# Face and brows [0:27]
#   jaw-cheek line: [0:17]  # 17
#   brows [17:27]           # 10
# Eyes, nose, mouth [27:67]
#   nose [27:36]            # 9
#   eyes [36:48]            # 12
#   mouth [48:68]           # 20

# left eye height: mark[38] - mark[40]
# right eye height: mark[43] - mark[47]
# mouth width: mark[48] - mark[54]
# mouth height: mark[62] -  mark[66]

class KernelCrop(object):
    def __call__(self, sample):
        crop = FaceCrop()
        marks = sample['landmarks']
        landmarks = np.array([marks[1], marks[8], marks[15],#jaw
                              marks[36], marks[45],#eyes
                              marks[48], marks[54]])#mouth
        return crop({'image': sample['image'], 'landmarks': landmarks})

class NoseTipCrop(object):
    def __call__(self, sample):
        crop = FaceCrop()
        return crop({'image': sample['image'], 'landmarks': sample['landmarks'][30:31]})

class MouthCrop(object):
    def __call__(self, sample):
        crop = FaceCrop()
        return crop({'image': sample['image'], 'landmarks': sample['landmarks'][48:]})

class EyesCrop(object):
    def __call__(self, sample):
        crop = FaceCrop()
        return crop({'image': sample['image'], 'landmarks': sample['landmarks'][36:48]})

class EyesMouthCrop(object):
    def __call__(self, sample):
        crop = FaceCrop()
        return crop({'image': sample['image'], 'landmarks': sample['landmarks'][36:]})

class FaceCrop(object):
    """Crop bounding box of face landmarks.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        left = -1
        right = 0
        top = -1
        bottom = 0
        for landmark in landmarks:
            if landmark[0] < left or left == -1:
                left = landmark[0]
            if landmark[1] < top or top == -1:
                top = landmark[1]

            if landmark[0] > right:
                right = landmark[0]
            if landmark[1] > bottom:
                bottom = landmark[1]
        new_w = np.int(right - left)
        new_h = np.int(bottom - top)

        top = np.int(top)
        left = np.int(left)

        image = image[top: (top + new_h), left: (left + new_w)]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #####
        image = image.transpose((2, 0, 1))
        # inverse would be = image.transpose((1, 2, 0))
        #####
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class ToNumpy(object):
    def __call__(self, sample):
        image = sample['image'].numpy().transpose(1, 2, 0)

        return {'image': image,
                'landmarks': landmarks.numpy()}

class ReplaceFaceBoundingBox(object):
    def __init__(self):
        self.face_coordinates = FaceCoordinates()

    def __call__(self, source, target_face):
        source_face_coor = self.face_coordinates(source)
        target_image = target_face['image']
        target_image = transform.resize(target_image, (source_face_coor['height'], source_face_coor['width']))
        source_image, source_landmarks = source['image'], source['landmarks']
        source_image[source_face_coor['top']:(source_face_coor['top'] + source_face_coor['height']),
                     source_face_coor['left']:(source_face_coor['left'] + source_face_coor['width']),
                     0:3] = target_image
        return {'image': source_image, 'landmarks': source_landmarks}

class ReplaceFace(object):
    def __init__(self):
        self.face_coordinates = FaceCoordinates()

    def _in_hull(self, w, h, hull, landmarks):
        for i in range(len(hull.vertices)):
            land = landmarks[hull.vertices[i]]
            next = landmarks[hull.vertices[(i+1)%len(hull.vertices)]]
            one = [w, h] - land
            two = next - [w, h]
            cross = one[0]*two[1] - one[1]*two[0]
            if cross > 0:
                return False
        return True

    def __call__(self, source, target_face):
        source_face_coor = self.face_coordinates(source)
        target_image_face, target_landmarks = target_face['image'], target_face['landmarks']
        hull =  ConvexHull(target_landmarks)
        source_image, source_landmarks = source['image'], source['landmarks']
        source_hull = ConvexHull(source_landmarks)

        corner = np.array([source_face_coor['left'], source_face_coor['top']])
        centers = np.array([target_landmarks[39], target_landmarks[42], target_landmarks[66]]) + corner

        replace_map = {}
        source_ave = source_image[0, 0, 0:3]
        target_ave = target_image_face[0, 0]

        print('replace (%d, %d)' % (source_face_coor['height'], source_face_coor['width']))
        count = 0
        for h in range(source_face_coor['height']):
            y = source_face_coor['top'] + h
            for w in range(source_face_coor['width']):
                x = source_face_coor['left'] + w
                center = [source_face_coor['left'] + source_face_coor['width']/2, source_face_coor['top'] + source_face_coor['height']/2]
                min_dist_center = math.sqrt((center[0] - x)**2 + (center[1] - y)**2)
                for c in range(len(centers)):
                    dist = math.sqrt((centers[c][0] - x)**2 + (centers[c][1] - y)**2)
                    if dist < min_dist_center:
                        center = centers[c]
                        min_dist_center = dist
                # Check if point is inside both faces convex hull.
                in_target_hull = True
                min_dist = sys.maxint #
                for i in range(len(hull.vertices)):
                    land = target_landmarks[hull.vertices[i]]
                    next = target_landmarks[hull.vertices[(i+1)%len(hull.vertices)]]
                    one = [w, h] - land
                    two = next - [w, h]
                    cross = one[0]*two[1] - one[1]*two[0]
                    if cross > 0:
                        in_target_hull = False
                        break
                    one_dist = math.sqrt(one[0]**2 + one[1]**2)
                    if one_dist < min_dist:
                        min_dist = one_dist

                if in_target_hull:
                    in_source_hull = self._in_hull(x, y, source_hull, source_landmarks)
                    if in_source_hull:
                        count += 1
                        dist_center = math.sqrt((center[0] - x)**2 + (center[1] - y)**2)
                        radius = (dist_center + min_dist)
                        value = math.sqrt(radius**2 - dist_center**2) # spherical
                        key = '{},{}'.format(h, w)
                        replace_map[key] = (radius - value) * source_image[y, x, 0:3] / radius + value * target_image_face[h, w] / radius
                        if count == 1:
                            source_ave = source_image[y, x, 0:3]
                            target_ave = target_image_face[h, w]
                        else:
                            source_ave = source_ave + source_image[y, x, 0:3]
                            target_ave = target_ave + target_image_face[h, w]

        source_ave /= count
        src_number_ave = (source_ave[0]+source_ave[1]+source_ave[2]) / 3
        target_ave /= count
        target_number_ave = (target_ave[0]+target_ave[1]+target_ave[2]) / 3
        diff_ave = target_number_ave - src_number_ave
        print('source_ave = %s', source_ave)
        print('target_ave = %s', target_ave)
        print('diff_ave = %s', diff_ave)
        for k in replace_map.keys():
            s = k.split(',')
            h = int(s[0])
            w = int(s[1])
            y = source_face_coor['top'] + h
            x = source_face_coor['left'] + w
            source_image[y, x, 0:3] = replace_map[k] - [diff_ave, diff_ave, diff_ave]

        print('replaced with iterations count = %d' % count)
        return {'image': source_image, 'landmarks': source_landmarks}
