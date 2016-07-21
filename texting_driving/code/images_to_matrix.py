
#Raj Agrawal 
#July 12, 2016 

"""
The following script reads in all images stored in a root directory and converts 
the images to an array.

Note: The root directory should just contain the desired images (o/w there
will be an error reading in a non-image) 
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread 
from scipy.ndimage.interpolation import zoom

def readImage(path, reduction_factor=.1):
    """
    - Reduces photo size by a factor of 1/reduction_factor
    - Converts to greyscale 
    """
    grey_image = imread(path, flatten=True)
    return zoom(grey_image, reduction_factor)

def to3DMatrix(paths, imgsize=(108, 192), num_frames):
    # Randomize?
    num_images = len(paths)
    images_by_time = np.zeros(shape=(108, 192, num_images)) #TODO fix so input imgsize
    for i, path in enumerate(paths):
        image = readImage(path)
        images_by_time[:, :, i] = image
        print('Finished Processing image ' + str(i))
    return images_by_time

def toMatrix(paths, imgsize=(108, 192), num_frames):
    # Randomize?
    num_images = len(paths)
    num_samples = num_images / num_frames 
    images_by_time = np.zeros(shape=(num_samples, 108, 192, num_frames))
    for sample_index in range(num_samples):
        index_in_array = sample_index * num_frames
        sample_paths = paths[index_in_array:(num_frames + index_in_array)]
        sample = to3DMatrix(sample_paths, imgsize, num_frames)
        images_by_time.concatenate(sample, axis=4)
        print('Finished Processing Sample ' + str(sample_index))
    return images_by_time

def makeLabels(file, paths):

#TODO: Need to make paths ordered so that images are in same 
#order 
def makePaths(folder_root):
    """
    - Returns the paths of all files in the folder_root 
    """
    return glob.glob(os.path.join(folder_root, '*'))

if __name__ == '__main__':

    path_to_images = raw_input('Enter Path to Images (no quotes):')
    
    # TODO: Make connnect to S3 

    # Read in data and save as a matrix 
    paths = makePaths(path_to_images)
    images_by_time = toMatrix(paths)
    np.save('./combined_data/images_by_time_mat', images_by_time)
    np.save('./combined_data/paths', paths)
    # Look at at image at time = 10
    # plt.imshow(images_by_time[:, :, 10], cmap='Greys_r')
