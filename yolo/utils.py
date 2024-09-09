import torch
import torchvision.transforms as transforms
from transforms import RandomHorizontalFlip, RandomRotation
from torchvision.transforms.functional import pil_to_tensor
from torch import nn

def scale(image):

    # Rescale the image to 416x416 pixels
    image = image.resize((416, 416))

    params = {"flipped": False, "angle": 0}   

    return image, params

def scale_and_normalize(image):

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    # Rescale the image to 416x416 pixels
    image = image.resize((416, 416))

    # normalize and convert to tensor
    image = pil_to_tensor(image)
    image = image.float() / 255.0
    image = normalize(image)

    params = {"flipped": False, "angle": 0}   

    return image, params

def scale_and_normalize_and_augment(image):

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    flip = RandomHorizontalFlip()
    rotate = RandomRotation(degrees=5)
    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    # Rescale the image to 416x416 pixels
    image = image.resize((416, 416))
    image, flipped = flip(image)
    image, angle = rotate(image)
    image = color_jitter(image)

    image = pil_to_tensor(image)
    image = image.float() / 255.0
    image = normalize(image)

    params = {"flipped": flipped, "angle": angle}   

    return image, params

def calc_iou(xc, yc, w, h, xg, yg, wg, hg):
    x1 = xc - w
    y1 = yc - h
    x2 = xc + w
    y2 = yc + h

    x3 = xg - wg
    y3 = yg - hg
    x4 = xg + wg
    y4 = yg + hg

    x5 = torch.max(x1, x3)
    y5 = torch.max(y1, y3)
    x6 = torch.min(x2, x4)
    y6 = torch.min(y2, y4)

    zero = torch.tensor(0)
    intersection = torch.max(zero, x6 - x5) * torch.max(zero, y6 - y5)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection

    return intersection / union

def calc_iou_from_points(x1, y1, x2, y2, x3, y3, x4, y4):
    # seperate ious function for points already calculated rather than having center points and width/height
    x5 = torch.max(x1, x3)
    y5 = torch.max(y1, y3)
    x6 = torch.min(x2, x4)
    y6 = torch.min(y2, y4)

    zero = torch.tensor(0)
    intersection = torch.max(zero, x6 - x5) * torch.max(zero, y6 - y5)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection

    return intersection / union