import torch
import torch.nn as nn
import torch.optim as optim

from imagenet import ImageNetDataset
from classifier import Darknet53
from solver import solver
from utils import transform_image_augment, scale

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

TRAINING_IMAGE_PATH = "C:\\Repos\\imagenet-1k\\training\\"
TRAINING_ANNOTATION_PATH = "C:\\Repos\\imagenet-1k\\labels\\synsets.txt"

VALIDATION_IMAGE_PATH = "C:\\Repos\\imagenet-1k\\validation\\"
VALIDATION_ANNOTATION_PATH = "C:\\Repos\\imagenet-1k\\labels\\val_labels.txt"

WEIGHTS_PATH = '.\classifer.weights'

def main():
    training_dataset = ImageNetDataset(TRAINING_IMAGE_PATH, TRAINING_ANNOTATION_PATH, transform_image_augment)
    validation_dataset = ImageNetDataset(VALIDATION_IMAGE_PATH, VALIDATION_ANNOTATION_PATH, scale)

    training = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Darknet53(weights_path=WEIGHTS_PATH)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_fn = nn.CrossEntropyLoss()

    solver(model, optimizer, loss_fn, training, validation, EPOCHS)

if __name__ == "__main__":
    main()

