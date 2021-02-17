import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
# NOTE this version of the program assumes the video is in the object-erase root directory
parser.add_argument('--video', type=str, default='', help='The video file you wish to process.')
args = parser.parse_args()

os.chdir('./Mask_RCNN')
subprocess.call(['python', 'process_video.py', '--video', args.video])
os.chdir('../sample-imageinpainting-HiFill/GPU_CPU')
video_path = '../../' + args.video
# the mask is stored in the root directory of the repo
mask_path = '../../mask.avi'
subprocess.call(['python', 'test2.py', '--video', video_path, '--mask', mask_path])