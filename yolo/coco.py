import os
import json
from PIL import Image
from torch.utils.data import Dataset
from copy import deepcopy
from anchors import ANCHORS
from utils import calc_iou
from torchvision.transforms.functional import pil_to_tensor
import torch
import math

class COCODataset(Dataset):
    """
    COCO dataset class for pytorch dataloader
    Responsible for loading images and labels from COCO dataset adn applying
    various transformations to the images and labels
    """
    def __init__(self, image_dir, label_dir, anchors=ANCHORS, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.anchors = anchors
        self.num_scale = len(anchors)
        self.num_anchors = len(anchors[0])
        self.grid_res = [32, 16, 8]
        self.resolution = 416

        self.image_paths = self._get_image_paths()
        self.labels = self._get_labels()

        self.max_objects = 50

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieve image and labels from dataset for given index number
        includes transformations and scaling of labels
        All images are resized to 416x416 pixels
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        _orginal_size = image.size

        if self.transform:
            # transform image
            image, params = self.transform(image)
        else:
            params = {"flipped": False, "angle": 0}

        if isinstance(image, Image.Image):
            _transformed_size = image.size
        elif isinstance(image, torch.Tensor):
            _transformed_size = tuple(image[0].size())

        image_id = image_path.split('\\')[-1].lstrip('0').split('.')[0]

        try:
            labels = deepcopy(self.labels[image_id])
        except KeyError:
            # remove image if it has no labels
            os.remove(image_path)
            print("Removed image with no labels")

            #return empty label for images that are missing it, dataset needs to be fixed
            return image, torch.zeros((self.max_objects, 9), dtype=torch.long)

        # scale bounding boxs according to the image transformation
        if _orginal_size != _transformed_size:
            labels = self._scale_label(labels, _orginal_size, _transformed_size)

        # flip bounding boxes if image was flipped
        if params["flipped"]:
            labels = self._flip_labels(labels, _transformed_size)

        # rotate bounding boxes if image was rotated
        if params["angle"]:
            labels = self._rotate_labels(labels, _transformed_size, params["angle"])

        tlabels = torch.zeros((self.max_objects, 7))

        # convert labels to tensor
        for i, label in enumerate(labels):

            if i == self.max_objects-1:
                # limit bounding boxes to set max
                break

            tlabels[i, 0] = label["category_id"]
            tlabels[i, 1:5] = torch.tensor(label["bbox"])
            tlabels[i, 5] = _transformed_size[0]
            tlabels[i, 6] = _transformed_size[1]

        targets = self._add_anchors2target(tlabels)

        return image, targets
    
    def _get_labels(self):
        """
        Open annotation file and load labels into json object
        """
        with open(self.label_dir, 'r') as f:
            labels = json.load(f)
        return labels

    def _get_image_paths(self):
        """
        Find all image paths in given directory
        """
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(".jpg"):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def _scale_label(self, label, original_size, transformed_size):
        """
        Scale the bounding boxes to match the transformed image size
        """
        width_ratio = transformed_size[0] / original_size[0]
        height_ratio = transformed_size[1] / original_size[1]

        for i, bbox in enumerate(label):

            label[i]["bbox"] = [int(bbox["bbox"][0] * width_ratio), 
                                int(bbox["bbox"][1] * height_ratio), 
                                int(bbox["bbox"][2] * width_ratio), 
                                int(bbox["bbox"][3] * height_ratio)]

        return label
    
    def _flip_labels(self, label, transformed_size):
        """
        Scale the bounding boxes to match a flipped image
        """
        for i, bbox in enumerate(label):

            label[i]["bbox"] = [int(transformed_size[0]-bbox["bbox"][0]-bbox["bbox"][2]), 
                                int(bbox["bbox"][1]), 
                                int(bbox["bbox"][2]), 
                                int(bbox["bbox"][3])]

        return label
    
    def _rotate_labels(self, label, transformed_size, angle):
        """
        Scale the bounding boxes to match a flipped image
        """
        for i, bbox in enumerate(label):
            x1, y1 = self._rotate_point((bbox["bbox"][0], bbox["bbox"][1]),(transformed_size[0]/2, transformed_size[1]/2), math.radians(-angle))
            x2, y2 = self._rotate_point((bbox["bbox"][0]+bbox["bbox"][2], bbox["bbox"][1]),(transformed_size[0]/2, transformed_size[1]/2), math.radians(-angle))
            x3, y3 = self._rotate_point((bbox["bbox"][0], bbox["bbox"][1]+bbox["bbox"][3]),(transformed_size[0]/2, transformed_size[1]/2), math.radians(-angle))
            x4, y4 = self._rotate_point((bbox["bbox"][0]+bbox["bbox"][2], bbox["bbox"][1]+bbox["bbox"][3]),(transformed_size[0]/2, transformed_size[1]/2), math.radians(-angle))

            x1 = min(x1, x3)
            y1 = min(y1, y2)
            x2 = max(x2, x4)
            y2 = max(y3, y4)

            w = x2 - x1
            h = y2 - y1
            
            label[i]["bbox"] = [int(x1), int(y1), int(w), int(h)]

        return label
    
    def _rotate_point(self, point, origin, angle):
        """
        Rotate a point around a given origin
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy
    
    def _add_anchors2target(self, in_targets):
        """
        converts raw annotations to targets with anchor information for use in loss function
        for each ground truth box, this function finds the anchor box with the highest iou and assigns it to
        be the target
        in_targets - (batch_size, targets, info) where info is (class, x, y, w, h, x_res, y_res)
        anchors - (scales, anchor) where anchor is (width, height)
        """

        # define sizes
        targets_size, _  = in_targets.size()

        # define output tensor
        out_targets = torch.zeros((targets_size, 9))

        obj = in_targets[..., 0]
        x_center = in_targets[..., 1] + in_targets[..., 3] // 2
        y_center = in_targets[..., 2] + in_targets[..., 4] // 2
        width = in_targets[..., 3] // 2
        height = in_targets[..., 4] // 2

        scale_num = torch.zeros((targets_size), dtype=torch.long)
        anchor_num = torch.zeros((targets_size), dtype=torch.long)
        x_grid = torch.zeros((targets_size), dtype=torch.long)
        y_grid = torch.zeros((targets_size), dtype=torch.long)

        for i, (xc, yc, w, h) in enumerate(zip(x_center.flatten(), y_center.flatten(), width.flatten(), height.flatten())):

            if xc == 0 and yc == 0 and w == 0 and h == 0:
                continue

            # clip center values to be within image resolution
            xc = torch.clamp(xc, 1, self.resolution-1)
            yc = torch.clamp(yc, 1, self.resolution-1)

            iou = torch.zeros((self.num_scale, self.num_anchors))
            for scale in range(self.num_scale):
                x_grid_center = xc // self.grid_res[scale] * self.grid_res[scale] + self.grid_res[scale] // 2
                y_grid_center = yc // self.grid_res[scale] * self.grid_res[scale] + self.grid_res[scale] // 2
                for anchor in range(self.num_anchors):
                    iou[scale, anchor] = calc_iou(xc, yc, w, h, x_grid_center, y_grid_center, self.anchors[scale][anchor][0] // 2, self.anchors[scale][anchor][1] // 2)
    
            scale_num[i], anchor_num[i] = torch.unravel_index(torch.argmax(iou), iou.shape)
            x_grid[i] = xc // self.grid_res[scale_num[i]] 
            y_grid[i] = yc // self.grid_res[scale_num[i]] 

        out_targets[..., 0] = obj
        out_targets[..., 1] = x_center
        out_targets[..., 2] = y_center
        out_targets[..., 3] = width
        out_targets[..., 4] = height
        out_targets[..., 5] = scale_num
        out_targets[..., 6] = anchor_num
        out_targets[..., 7] = x_grid
        out_targets[..., 8] = y_grid

        # all interger values anyway
        out_targets = out_targets.long()

        # out_target (batch_size, targets, info) where info is (obj, x, y, w, h, scale_num, anchor_num, x_grid, y_grid)
        return out_targets