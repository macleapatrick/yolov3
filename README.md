Files to train a yolo (you only look once) version 3 object detection and localization convolutional neural network from scratch developed in Pytorch.

OVERVIEW:

classifier\classifer.py - This script implements a custom PyTorch model based on the Darknet-53 architecture. The model is composed of convolutional and residual blocks for deep feature extraction.

classifier\ImageNetDataset.py - This script defines a custom PyTorch dataset subclass, ImageNetDataset, designed for loading and processing images and labels for the ImageNet dataset. It is structured to handle both training and validation data.

classifier\labels.py - Dict object for mapping numerical class number to textual label

classifier\main.py - Trains a Darknet53 classifier on the ImageNet dataset using PyTorch. It sets up data loading, model initialization, optimization, and loss calculation, and uses a solver function to handle the training process.

classifier\solver.py - Defines a solver function that manages the training and evaluation loop for a neural network model using PyTorch. It leverages mixed precision training (via torch.cuda.amp) for efficiency and includes functionality for saving model weights during training.

classifier\testcases.py - Just some test cased used during devlopemnt and debugging

classifier\utils.py - Utility functions, also have image augmentation functions.
  
yolo\anchor.py - Hand selected priors for use in scaling relative outsputs of the yolo model

yolo\calculate_anchors.py - K-means clustering of bounding boxes across coco training dataset

yolo\classweights.py - Number of occurances of each class in the coco training dataset, used to scale the loss calculations when training to account for class inbalance

yolo\coco.py - Defines the COCODataset class, which is designed to load and process the COCO dataset for object detection tasks using PyTorch. It handles image loading, label processing, and augmentation transformations, and also prepares targets for use in loss calculation. This dataset class is designed to handle the complexity of object detection training, including data augmentation, scaling, and anchor-based bounding box prediction.

yolo\labels.py -Python dict object mapping numerical labels to textual labels.

yolo\livecamerafeed.py -Quick fun implimention to run a yolo model through a webcam.

yolo\loss.py - A custom loss function for training YOLOv3 models in PyTorch. It handles the calculation of bounding box regression, objectness, and classification losses, and supports multiple anchor boxes and grid scales. This class is essential for training YOLOv3 models, handling the complex multi-task nature of object detection (localizing objects, classifying them, and determining their presence).

yolo\main.py - Sets up and executes the training process for a YOLOv3 object detection model using the COCO dataset. It handles dataset loading, model initialization, and training using the YOLOv3 loss function. This script provides the full training pipeline for the YOLOv3 model, utilizing the COCO dataset and handling all aspects from data loading to model optimization and weight management.

yolo\model.py - Implements the YOLOv3 (You Only Look Once, Version 3) model using PyTorch. It integrates a feature extractor from Darknet53 and defines custom layers for multi-scale object detection. The model predicts bounding boxes, objectness scores, and class probabilities at three different scales (13x13, 26x26, and 52x52 grids). This model is designed for efficient multi-scale object detection using predefined anchor boxes, and it is capable of making accurate predictions at various scales. The convert2bb() function is included for post-processing model outputs into usable bounding boxes with confidence scores.

yolo\solver.py - Defines a solver function for training and evaluating the model. It handles the training process across multiple epochs, calculates losses at different detection scales, and provides periodic updates on training progress.

yolo\testcases.py - Testing and debugging model and training implimentation

yolo\transforms.py - Subclasses of some pytorch image transformations to return additional details to augment bounding boxes as well

yolo\utils.py - Helper functions including image transforms/augments and iou functions for bounding box calculation

yolo\yolo.py - Defines a Yolo class that encapsulates the YOLOv3 model for running object detection on input images. It handles preprocessing, model inference, and post-processing (such as non-max suppression and bounding box drawing). Provides a full pipeline for running object detection on images, from preprocessing to drawing bounding boxes with class labels. The model uses a pre-trained YOLOv3 and efficiently performs inference, leveraging GPU acceleration if available.


