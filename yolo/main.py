import torch
import torch.nn as nn
import torch.optim as optim
import os

from coco import COCODataset
from model import Yolov3
from loss import YOLOv3Loss
from solver import solver
from utils import scale_and_normalize_and_augment
from anchors import ANCHORS

BATCH_SIZE = 8
LEARNING_RATE = 0.00003
EPOCHS = 50

TRAINING_IMAGE_PATH = "C:\\Repos\\coco\\coco2017\\train2017\\"
TRAINING_ANNOTATION_PATH = "C:\\Repos\\coco\\coco2017\\annotations\\annotations_train.json"

CLASSIFIER_WEIGHTS_PATH = '.\\classifer.weights'
YOLO_WEIGHTS_PATH = '.\\yolov3.weights'


def main():
    training_dataset = COCODataset(TRAINING_IMAGE_PATH, TRAINING_ANNOTATION_PATH, transform=scale_and_normalize_and_augment)

    training = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    model = Yolov3(weights_path=YOLO_WEIGHTS_PATH, darknet_weights_path=CLASSIFIER_WEIGHTS_PATH)

    if os.path.isfile(YOLO_WEIGHTS_PATH):
        model.load_weights()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_fn = YOLOv3Loss(ANCHORS, BATCH_SIZE)

    solver(model, optimizer, loss_fn, training, None, EPOCHS)

if __name__ == "__main__":
    main()

