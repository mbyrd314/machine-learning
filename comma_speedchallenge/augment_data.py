# Script to generate new training frames by flipping and adjusting the brightness
# in the given training frames

import os
import skimage
from skimage import exposure, io
import numpy as np
# This should be called from the ../speedchallenge/data directory

def flip_vert(image):
    return np.flipud(image)

def flip_horiz(image):
    return np.fliplr(image)

def adjust_brightness(image, gamma):
    return exposure.adjust_gamma(image, gamma)

if __name__ == '__main__':
    path = 'train_frames'
    horiz_flip_path = 'horiz_flip_frames'
    vert_flip_path = 'vert_flip_frames'
    both_flip_path = 'both_flip_frames'
    brighter_path = 'brighter_frames'
    darker_path = 'darker_frames'

    image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    count = 0
    for image_path in image_paths:
        image = io.imread(image_path)
        horiz_flip_image = flip_horiz(image)
        vert_flip_image = flip_vert(image)
        both_flip_image = flip_horiz(flip_vert(image))
        brighter_image = adjust_brightness(image, .75) # This gamma value is arbitrary
        darker_image = adjust_brightness(image, 1.25) # This gamma value is arbitrary
        io.imsave(f'{horiz_flip_path}/frame{count}.jpg', horiz_flip_image)
        io.imsave(f'{vert_flip_path}/frame{count}.jpg', vert_flip_image)
        io.imsave(f'{both_flip_path}/frame{count}.jpg', both_flip_image)
        io.imsave(f'{brighter_path}/frame{count}.jpg', brighter_image)
        io.imsave(f'{darker_path}/frame{count}.jpg', darker_image)
        count += 1
