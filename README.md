# Object-Erase
## Detecting and erasing objects in videos using machine learning.

Inspired by [this video](https://www.youtube.com/watch?v=U7LudBS3bS4), this project is an open-source attempt at recreating and generalizing such effects. 

Mask-RCNN is used to detect objects in video and generate a mask video file which is then inputted into HiFill generative image inpainting to erase objects and fill in the gaps.

## Requirements

- Python 3.7.7
- opencv-python 4.5.1
- numpy
- Pillow
- cython
- matplotlib
- scikit-image
- tensorflow>=1.3.0 (must be tensorflow 1 as there are compatibility issues with 2)
- keras>=2.0.8
- opencv-python
- h5py
- imgaug
- IPython[all]


The software was devloped and tested with Python 3.7.7, tensorflow 1.15.3, opencv-python 4.5.1, and numpy 1.20.1. See Mask_RCNN/requirements.txt for a full list 
of python dependencies, and note that the required dependencies for HiFill form a subset of this list. All coding was done on Fedora 32. An Nvidia GPU 
is not required, although it would be very helpful. Development of this project began in the summer of 2019 when I had no tensorflow-compatible GPU. As one can 
imagine, testing took forever.

### MS Coco Api

Mask_RCNN requires the MS Coco dataset api to run properly. The coco api is stored as a submodule within Mask_RCNN. 
To install the api,
`cd Mask_RCNN/coco/PythonAPI`

`make`

`python setup.py build_ext install`

## Usage

`python main.py --video <video_name>`  should run the software and generate a final video also in the root directory. 
**Note:** `main.py` assumes the video file is stored in the root directory of the repository. If a path to a video file stored elsewhere
in your system is given, there are no guarantees the software will execute properly. There is one exception to this rule:

### main.py doesn't work

During testing a strange error was encountered regarding opencv's videowriter (see issues). After much digging it was determined that opencv-python sometimes does 
not cooperate when being run outside of the Mask_RCNN and HiFill directories. If this is the case for you, there is still hope. You can replicate `main.py` manually:

`cd Mask_RCNN`

`python process_video.py --video <video_file>`

`cd ../sample-imageinpainting-HiFill/GPU_CPU`

`python test2.py --video <video_file> --mask <mask_file>`

Note that the mask file outputted by `process_video.py` always saves to the root directory of the repository. Manually running the software like this has the 
added benefit of allowing video files to be inputted from anywhere in your system (as opposed to being stored in the root directory).

## Configuring Mask_RCNN to detect only certain objects.

Suppose you only want to detect and erase cars, people, boats, or animals. The program can be quickly modified work this way.
Under `Mask_RCNN` open `visualize_cv2.py` with a text editor. On line 30 you will see a list called `class_names` which contains 
every type of object Mask_RCNN is able to detect. Suppose you only want to detect vehicles. Then you can define a sublist of 

just the vehicles in `class_names` such as the one on line 48. Next, in the function definition for `display_mask()` (line 64), on line 82 within the for loop you 
will see a commented `if` conditional statement. This conditional is an example of checking whether a detected object is a vehicle. The subsequent two lines 
within the loop should be indented into the body of the conditional statement, and the conditional statement uncommented. This method can be modified to work with 
any subset of objects within `class_names`. 

