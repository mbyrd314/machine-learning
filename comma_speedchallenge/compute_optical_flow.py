import os
import cv2
import numpy as np

# Converts all training frames and augmented data frames into optical flow images
# to be used as inputs to neural network. Calculating all of this beforehand to save
# time in actual training function

if __name__ == '__main__':
    root_dir = '/home/michael/speedchallenge/data'
    frame_dirs = ['train_frames', 'horiz_flip_frames', 'vert_flip_frames', 'both_flip_frames', 'darker_frames', 'brighter_frames']
    input_dirs = ['train_inputs', 'horiz_flip_inputs', 'vert_flip_inputs', 'both_flip_inputs', 'darker_inputs', 'brighter_inputs']
    #image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(path, f))]
    count = 0
    num_frames = 20400
    for i, dir in enumerate(frame_dirs):
        path = os.path.join(root_dir, dir)
        input_dir = os.path.join(root_dir, input_dirs[i])
        #print(f'path: {path}, input_dir: {input_dir}')
        #image_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for j in range(num_frames-1):
            frame_path1 = os.path.join(path, f'frame{j}.jpg')
            frame_path2 = os.path.join(path, f'frame{j+1}.jpg')
            #print(f'frame_path1: {frame_path1}, frame_path2: {frame_path2}')
            frame1 = cv2.imread(frame_path1)
            frame2 = cv2.imread(frame_path2)
            hsv = np.zeros_like(frame1)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 1] = 255 # Saturation is max
            hsv[..., 0] = ang*180/np.pi/2 # Converting hue to the right range
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            input_path = os.path.join(input_dir, f'input{j}.png')
            cv2.imwrite(input_path, rgb)
            if j % 1000 == 999:
                print(f'i: {i}, j: {j}')
