# Author: Raj Agrawal 

# The following script reads in all images stored in a root directory and converts 
# the images to an array.

# Run $ffmpeg -i NAME.MOV -r 20 ../data/train/images/image_sequence%d.jpeg from
# code folder to generate images in the training images folder

# Add more documentation, convert to pep8 

from __future__ import division 

import os
import glob
import numpy as np
import pandas as pd 

from scipy.misc import imread 
from scipy.ndimage.interpolation import zoom

def readImage(path, reduction_factor=.1):
    """
    - Reduces photo size by a factor of 1/reduction_factor
    - Converts to greyscale 
    """
    grey_image = imread(path, flatten=True)
    return zoom(grey_image, reduction_factor)

def toMatrix(paths, num_frames, imgsize=(108, 192)):
    """
    - num_samples x num_frames x length x width array 
    """
    num_images = len(paths)
    num_samples = int(num_images / num_frames)
    images_by_time = np.zeros(shape=(num_samples, num_frames, imgsize[0], imgsize[1])) 
    for sample_index in range(num_samples):
        index_in_array = sample_index * num_frames
        sample_paths = paths[index_in_array:(num_frames + index_in_array)]
        sample = to3DMatrix(sample_paths, imgsize)
        images_by_time[sample_index, :, :, :] = sample 
        print('Finished Processing Sample ' + str(sample_index))
    return images_by_time

def to3DMatrix(paths, imgsize=(108, 192)):
    """
    - len(paths) x length x width array 
    """
    num_images = len(paths)
    images_by_time = np.zeros(shape=(num_images, imgsize[0], imgsize[1])) 
    for i, path in enumerate(paths):
        image = readImage(path)
        images_by_time[i, :, :] = image
        print('Finished Processing image ' + str(i))
    return images_by_time

def toDurations(stopped_times_list):
    to_seconds = []
    for time in stopped_times_list:
        print(time)
        mins, secs = time.split(':') 
        if mins == '':
            mins = 0
        to_sec = int(mins) * 60 + int(secs) 
        to_seconds.append(to_sec)
    num_times = len(to_seconds)
    from_zero = [0] + to_seconds[0:(num_times - 1)]
    time_diffs = np.array(to_seconds) - np.array(from_zero)
    return time_diffs

def makeLabels(file_label, samps_per_sec=2):
    """
    - Frames happen every .05 seconds, one sample corresponds w/ 10 frames or
      2 samples / seconds
    """
    labels_by_time = pd.read_csv(file_label, header=None)
    to_durations = toDurations(list(labels_by_time[1]))
    labels_per_phase = list(labels_by_time[0])
    num_samps_per_sec = to_durations * samps_per_sec
    sample_labels = []
    for i, label in enumerate(labels_per_phase):
        num_samps = num_samps_per_sec[i]
        sample_labels += list(np.tile(label, num_samps))
    return np.array(sample_labels)

def makePaths(folder_root):
    """
    - Returns the paths of all files in the folder_root 
    """
    return glob.glob(os.path.join(folder_root, '*'))

def makeOrderedPaths(folder_root, num_pics):
    """
    - Returns the paths of all files in the folder_root but keeping frames in 
      temporal order 
    """
    paths = [folder_root + '/image_sequence' + str(i) + '.jpeg' for i in range(1, num_pics + 1)]
    return paths

if __name__ == '__main__':

    path_to_images = '../data/train/images'
    path_to_lables = '../data/train/video_labels.csv'
    
    # Read in video data/labels
    labels = makeLabels(path_to_lables, samps_per_sec=2) 
    num_pics = len(makePaths(path_to_images))
    paths = makeOrderedPaths(path_to_images, num_pics)
    images_by_time = toMatrix(paths=paths, num_frames=10)
    
    # Shuffle data
    # Check to make sure this matches images_by_time --> might need to pad ends w/ extra labels 
    num_samples = images_by_time.shape[0]
    indcs = np.arange(num_samples)
    np.random.shuffle(indcs)
    images_by_time = images_by_time[indcs]
    labels = labels[indcs]

    # Save in data folder 
    np.save('../data/train/images_by_time_mat', images_by_time)
    np.save('../data/train/labels', labels)
