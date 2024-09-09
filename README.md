Files to train a yolo (you only look once) version 3 object detection and localization convolutional neural network from scratch developed in Pytorch.

Overview:
  classifier:
    - Used to define and train the classifer backbone used in the parent yolo network.  Trained on ImageNet 2012 dataset with standard data augmentation
  yolo:
    - Definations to train and run the yolo network on images and video feeds.  Set up to train on the COCO 2017 datatset. 
