import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pil_to_tensor

def transform_image_augment(image):

    rotate = transforms.RandomRotation(15)
    flip = transforms.RandomHorizontalFlip()
    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    # Rescale the image to 416x416 pixels
    image = image.resize((256, 256))
    image = rotate(image)
    image = flip(image)
    image = color_jitter(image)

    image = pil_to_tensor(image)
    image = image.float() / 255.0
    image = normalize(image)

    return image

def scale(image):

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    # Rescale the image to 256x256 pixels
    image = image.resize((256, 256))

    image = pil_to_tensor(image)
    image = image.float() / 255.0
    image = normalize(image)

    return image

def one_hot_encode(labels, num_classes=1000):
    one_hot_labels = torch.zeros(labels.size(0), num_classes)
    one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
    return one_hot_labels