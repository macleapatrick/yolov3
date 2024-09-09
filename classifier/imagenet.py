import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class ImageNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.labels = self._get_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if 'training' in image_path:
            folder_name = image_path.split('\\')[-2]
            label = int(self.labels.index(folder_name))

        elif 'validation' in image_path:
            label = int(self.labels[idx])

        return image, label
    
    def _get_labels(self):
        with open(self.label_dir, 'r') as f:
            labels = f.readlines()
            labels = [label.strip() for label in labels]
        return labels

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(".JPEG"):
                    image_paths.append(os.path.join(root, file))
        return image_paths