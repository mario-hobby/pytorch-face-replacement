"""Create a sample face landmarks dataset.

Adapted from dlib/python_examples/face_landmark_detection.py
See this file for more explanation.

Download a trained facial shape predictor from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
import sys
import dlib
import glob
import csv
import argparse
from skimage import io

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')
num_landmarks = 81

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Path to data dir. Like: 'data/faces/subdir'")
# read arguments from the command line
args = parser.parse_args()

dir = args.input_dir# sys.argv[1] # '../data/faces/subdir'
print('dir=', dir)

images_regex = '%s/*.jpg' % dir
print('Will read images in: %s' % images_regex)
landmarks_filename = '%s/face_landmarks.csv' % dir
print('Will write landmarks to: %s' % landmarks_filename)

with open(landmarks_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    header = ['image_name']
    for i in range(num_landmarks):
        header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]

    csv_writer.writerow(header)

    for f in glob.glob(images_regex):
        print('file: %s' % f)
        img = io.imread(f)
        dets = detector(img, 1)  # face detection

        # ignore all the files with no or more than one faces detected.
        if len(dets) == 1:
            print('detected face')
            filename = f.split('/')[-1]
            row = [filename]

            d = dets[0]
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            for i in range(num_landmarks):
                part_i_x = shape.part(i).x
                part_i_y = shape.part(i).y
                row += [part_i_x, part_i_y]

            csv_writer.writerow(row)
