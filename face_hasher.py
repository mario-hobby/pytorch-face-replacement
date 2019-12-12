import sys
import random
import string
import time
import math
import os
from scipy.spatial import ConvexHull
import torch

class FaceHasher:

    @staticmethod
    def to_array(landmarks):
        array = []
        for i in range(len(landmarks)):
            array.append(int(landmarks[i][0]))
            array.append(int(landmarks[i][1]))
        return array

    @staticmethod
    def dist(a, b):
        return abs(a[1] - b[1])

    # left eye height: mark[38] - mark[40]
    # right eye height: mark[43] - mark[47]
    # mouth width: mark[48] - mark[54]
    # mouth height: mark[62] -  mark[66]
    @staticmethod
    def hash(resolution, face):
        result = [str(resolution)]
        left_eye = int(FaceHasher.dist(face[38], face[40])) / resolution
        result.append(str(left_eye))
        right_eye = int(FaceHasher.dist(face[43], face[47])) / resolution
        result.append(str(right_eye))
        mouth_height = int(FaceHasher.dist(face[66], face[62])) / resolution
        result.append(str(mouth_height))
        return string.join(result, ',')

    @staticmethod
    def mass_center(resolution, landmarks):
        x = y = 0
        result = [str(resolution)]
        for i in range(len(landmarks)):
            x += int(landmarks[i][0] / resolution)
            y += int(landmarks[i][1] / resolution)
        result.append(str(x / resolution))
        result.append(str(y / resolution))
        return string.join(result, ',')
