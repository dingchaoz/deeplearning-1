
# Taken mostly from https://jessesw.com/Deep-Learning/

from scipy.ndimage import convolve, rotate
import numpy as np

def random_image_generator(image_stack):
    '''
    This function will randomly translate and rotate an image, producing a new, altered version as output.
    '''
    num_frames = image_stack.shape[0]
    length = image_stack.shape[1]
    width = image_stack.shape[2]
    # Create our movement vectors for translation first. 
        
    move_up = [[0, 1, 0],
               [0, 0, 0],
               [0, 0, 0]]
        
    move_left = [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]
        
    move_right = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]]
                                   
    move_down = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]
        
    # Create a dict to store these directions in.
        
    dir_dict = {1:move_up, 2:move_left, 3:move_right, 4:move_down}
        
    # Pick a random direction to move.
        
    direction = dir_dict[np.random.randint(1,5)]
        
    # Pick a random angle to rotate (30 degrees clockwise to 30 degrees counter-clockwise).
        
    angle = np.random.randint(-30,31)
        
    # Move the random direction and change the pixel data back to a 2D shape.
    new_image = np.zeros(shape=(num_frames,length, width))
    for i, image in enumerate(image_stack):
        moved = convolve(image.reshape(length,width), direction, mode = 'constant')
        # Rotate the image
        rotated = rotate(moved, angle, reshape = False)
        new_image[i, :, :] = rotated
    return new_image
