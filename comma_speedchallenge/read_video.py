# Function to convert the training mp4 file into individual frames
import cv2
vidcap = cv2.VideoCapture('/home/michael/speedchallenge/data/train.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print(f'Read a new frame: {success}')
  count += 1
