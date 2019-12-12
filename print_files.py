# Simply print all the file names.
import glob

dir = '../data/faces/'
images_regex = '%s/*.png' % dir
print('Will read images in: %s' % images_regex)
for f in glob.glob(images_regex):
    print('%s' % f)
