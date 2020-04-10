import cv2
import os
# Helper function to convert the augmented data frames that I generated into video
# files so that I can compare them to the original training video. I'm not sure why
# the frame rate, which I set to 20 which the training video apparently used, is so
# much faster than the training video
def make_video(root_dir, frame_dir):
    video_name = f'{frame_dir}.avi'
    frame_dir = os.path.join(root_dir, frame_dir)
    video_path = os.path.join(frame_dir, video_name)
    images = [img for img in os.listdir(frame_dir)]
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, channels = frame.shape

    video = cv2.VideoWriter(video_path, 0, 20, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(frame_dir, image)))

    video.release()

if __name__ == '__main__':
    root_dir = '/home/michael/speedchallenge/data'
    frame_dirs = ['horiz_flip_frames', 'vert_flip_frames', 'both_flip_frames', 'darker_frames', 'brighter_frames']
    for frame_dir in frame_dirs:
        make_video(root_dir, frame_dir)
