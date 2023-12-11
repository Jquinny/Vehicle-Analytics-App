# Vehicle-Analytics-App

application for analyzing vehicles in cctv roadway footage, gathering vehicle data such
as timestamp of appearance, class and class confidence, and entry/exit directions

#### Contributions
Joey Quinlan - Backend video processing and active learning (headless mode)

Lingrui Zhou - GUI functionality

## Repository Setup

You can get the project code either by cloning the repository or by
downloading it as a ZIP file by clicking on the green Code button

if cloning via ssh, run

`git clone git@github.com:Jquinny/Vehicle-Analytics-App.git`

in
a terminal of your choice. Make sure to run it in the directory where you want
the repository downloaded to.

## Python Environment Setup

First, make sure you have a version of python >= 3.9.X (3.10 is what this has
been tested on so far)

**Note:** use `python3` instead of `python` if on linux/mac

1. run `python -m venv env` from the root directory of this repository. You should end up with an env/ directory inside the vehicle-analytics-app directory
2. activate the environment by running `.\env\Scripts\activate.bat ` if on windows or `source env/bin/activate` if on linux or mac
3. run `python -m pip install -r requirements.txt` to install all dependencies into the virtual environment (MAKE SURE THE ENVIRONMENT IS ACTIVATED FIRST)
4. install PyTorch dependencies using one of the following commands
   - Windows: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
   - Linux: `pip3 install torch torchvision torchaudio`
   - Mac: `pip3 install torch torchvision torchaudio` **Note:** Mac does not support CUDA, so no gpu access (would not recommend running on a mac)
5. install src/ directory as a python package (run this from the root directory of the repository): `pip install --editable .`

## Running Instructions

To run the GUI application, enter the following command in a terminal at the root of the repository:

`python src/TruckUI.py`

To run the headless processing algorithm on a video, enter the following command in a terminal at the root of the repository:

`python src/process_video.py <absolute path to video> <command line arguments>`

where the available arguments can be seen by running:

`python src/process_video.py -h`

**Note:** if there are plans to run the processing algorithm on a ton of videos from the same camera view, follow the suggested folder structure in the videos/ [README](https://github.com/Jquinny/Vehicle-Analytics-App/tree/main/videos) file. This allows for re-use of user defined ROI polygon and direction coordinates across the videos.

## Model Training

There is a training script in the src/ directory called train.py. This training script allows the user to train yolov8s, yolov8m, and yolov8l object detectors using a custom data set. Run it with the following:

`python src/train.py <absolute path to dataset folder> <command line arguments>`

where the available arguments can be seen by running:

`python src/train.py -h`

Once a model has been trained, it will end up in the models/ directory, ready to use immediately with the application. The [README](https://github.com/Jquinny/Vehicle-Analytics-App/blob/main/models/README.md) in the models/ directory contains more information on the structure of things

**Notes:**

- the dataset should be in the yolov8.txt format
- For windows users, delimit backslashes when entering the absolute path to the dataset, so C:\Users\user\dataset should be entered as C:\\\Users\\\user\\\dataset
- the script is not well tested and may have varying success on different platforms, so if any issues arise add them as an issue on github
- support for classifier training may be added in the future, and the implementation of the training script itself may be updated to increase ease of use
