Cyclist Detection with YOLOv5

This project is an implementation of the YOLOv5 architecture for cyclist detection. It uses a modified version of the Tsinghua-Daimler Cyclist Detection Benchmark Dataset and is structured for easy understanding and replication.

Project Structure

- src
  --hyperparameters.py: Contains all the hyperparameters. This is the only file that needs to be adapted to modify the experiment.
  --visualize.py: Provides the functions to display the dataset distribution and results of the experiment.
- data
  --Contains a modified version of the Tsinghua-Daimler Cyclist Detection Benchmark Dataset. This version focuses on a single class: cyclists, and has been labelled in the YOLOv5 format.

- main.ipynb
  --The primary Jupyter notebook where functions are called, YOLOv5 training is executed, and visualizations are displayed.

- sort.ipynb
  --This notebook demonstrates data sorting to take a subsample. It's meant for demonstration purposes only. It is not set up to run in this project folder due to invalid paths and the absence of the complete dataset.

- yolov5
  --This folder contains the Ultralytics YOLOv5 code. I have made updates to the following:
  --Updated `cyclists.yaml` to set relative paths to datasets.
  --Additional hyperparameter .yaml files designed to experiment with incremental increases in iou_t to study architecture accuracy impact.
  --The remaining contents of this folder originate from Ultralytics and need to be cited accordingly.

Credits
--The YOLOv5 code was sourced from the Ultralytics repository: [Ultralytics YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5.git)
--Dataset: Tsinghua-Daimler Cyclist Detection Benchmark Dataset (Li et al., 2016)

Setup

All of the code can be run from main.ipynb.

Uncomment the "#%pip install -r requirements.txt" before running the code from the main notebook

Uncomment the code to install YOLOv5:
$ git clone https://github.com/ultralytics/yolov5...
