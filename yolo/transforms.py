import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch import Tensor
import torch

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """
    Horizontal flip but returns a flag indicating if the image was flipped
    """
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False
    
class RandomRotation(transforms.RandomRotation):
    """
    Random rotation but returns the angle of rotation
    """
    def __init__(self, degrees=0):
        super().__init__(degrees)

    def forward(self, img):
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.interpolation, self.expand, self.center, fill), angle