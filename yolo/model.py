import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import Darknet53, ConvModule
from utils import calc_iou_from_points
from anchors import ANCHORS
import os
import time

class Yolov3(nn.Module):
    def __init__(self, weights_path=None, darknet_weights_path=None, num_classes=90, num_bb=3):
        super(Yolov3, self).__init__()
        self.num_bb = num_bb
        self.output_per_bb = 5 + num_classes
        self.weights_path = weights_path
        self.grid = torch.tensor([13, 26, 52])
        self.resolution = 416
        self.anchors = ANCHORS

        if not os.path.isfile(weights_path):
            if os.path.isfile(darknet_weights_path):
                # first time initializing yolo, load darknet53 weights serperately
                self.darknet53 = Darknet53(weights_path=darknet_weights_path, load_weights=True, classifier_only=False)
            else:
                # Just seeing what happens if the convnet is freshly initialized
                self.darknet53 = Darknet53(classifier_only=False)
        else:
            # weights will be loaded as part of wider yolo model
            self.darknet53 = Darknet53(classifier_only=False)

        self.head1 = nn.Sequential(
            ConvModule(384, 128, 1, 1, 0), 
            ConvModule(128, 256, 3, 1, 1),
            ConvModule(256, 128, 1, 1, 0), 
            ConvModule(128, 256, 3, 1, 1),
            ConvModule(256, 128, 1, 1, 0)
            )
        
        self.head2 = nn.Sequential(
            ConvModule(768, 256, 1, 1, 0), 
            ConvModule(256, 512, 3, 1, 1),
            ConvModule(512, 256, 1, 1, 0), 
            ConvModule(256, 512, 3, 1, 1),
            ConvModule(512, 256, 1, 1, 0)
            )
        
        self.head3 = nn.Sequential(
            ConvModule(1024, 512, 1, 1, 0), 
            ConvModule(512, 1024, 3, 1, 1),
            ConvModule(1024, 512, 1, 1, 0), 
            ConvModule(512, 1024, 3, 1, 1),
            ConvModule(1024, 512, 1, 1, 0)
            )

        self.upscale2 = nn.Sequential(
            ConvModule(256, 128, 1, 1, 0), 
            nn.Upsample(scale_factor=2)
            )
        
        self.upscale3 = nn.Sequential(
            ConvModule(512, 256, 1, 1, 0), 
            nn.Upsample(scale_factor=2)
            )
        
        self.output1 = nn.Sequential(
            ConvModule(128, 256, 3, 1, 1),
            nn.Conv2d(256, self.num_bb * self.output_per_bb, 1, 1, 0)
            )
        
        self.output2 = nn.Sequential(
            ConvModule(256, 512, 3, 1, 1),
            nn.Conv2d(512, self.num_bb * self.output_per_bb, 1, 1, 0)
            )

        self.output3 = nn.Sequential(
            ConvModule(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.num_bb * self.output_per_bb, 1, 1, 0)
            )
        
        if not os.path.isfile(weights_path):
            # first time initializing yolo, save parameters after initializing
            self.save_weights()

    def forward(self, x):
        branch1, branch2, branch3 = self.darknet53(x)

        branch3 = self.head3(branch3)
        branch2 = self.head2(torch.cat((self.upscale3(branch3), branch2), dim=1))
        branch1 = self.head1(torch.cat((self.upscale2(branch2), branch1), dim=1))

        output1 = self.output1(branch1)
        output2 = self.output2(branch2)
        output3 = self.output3(branch3)

        #shape feature map in a format easier to work with for loss/inference
        output1 = output1.contiguous().view(-1, self.num_bb, self.output_per_bb, 52, 52).permute(0, 1, 3, 4, 2)
        output2 = output2.contiguous().view(-1, self.num_bb, self.output_per_bb, 26, 26).permute(0, 1, 3, 4, 2)
        output3 = output3.contiguous().view(-1, self.num_bb, self.output_per_bb, 13, 13).permute(0, 1, 3, 4, 2)
        
        return output1, output2, output3
    
    def load_weights(self, weights_path=None):

        if self.weights_path:
            weights = self.weights_path
        else:
            weights = weights_path

        # Load the pre-trained weights
        if os.path.isfile(weights):
            self.load_state_dict(torch.load(weights))
        
    def save_weights(self, weights_path=None):

        if self.weights_path:
            weights = self.weights_path
        else:
            weights = weights_path

        torch.save(self.state_dict(), weights)

    def convert2bb(self, in_predictions, obj_threshold=0.4, iou_threshold=0.4):
        # converts the absolute predictions transformed by the loss function to bounding boxes
        predictions = None
        field_length = 8

        for prediction_scale in in_predictions:
            # Elimate predictions under 
            indices = torch.nonzero(prediction_scale[..., 4] > obj_threshold, as_tuple=True)
            p = torch.zeros((indices[0].size(0), field_length))

            # form output matrix
            for i in range(len(indices[0])):
                a = prediction_scale[indices[0][i], indices[1][i], indices[2][i], indices[3][i]]
                p[i, 0] = a[0] - a[2] # x1
                p[i, 1] = a[1] - a[3] # y1
                p[i, 2] = a[0] + a[2] # x2
                p[i, 3] = a[1] + a[3] # y2
                p[i, 4] = a[4] # objectness confidence
                p[i, 5] = torch.argmax(a[5:]) + 1 # object class
                p[i, 6] = a[torch.argmax(a[5:])+5] # class confidence
                p[i, 7] = a[4] * p[i, 6] # combined confidence

            if predictions is None:
                predictions = p
            else:
                predictions = torch.cat((predictions, p), dim=0)
        
        if predictions.numel() == 0:
            return None
        
        # perform NMS
        torch.stack(sorted(predictions, key=lambda a: a[7], reverse=True))
        predictions = predictions.detach()

        for i, prediction in enumerate(predictions):

            if i == len(predictions):
                break
            elif torch.all(prediction == torch.zeros((field_length))):
                continue    

            #if prediction[7] < 0.4:
                # drop predictions with low combined confidence, usually from objects with high objectness but low class confidence
                #predictions[i] = torch.zeros((field_length))
                #continue

            for j, sub in enumerate(predictions[i+1:]):
                # same object class
                if prediction[5] == sub[5]:
                    if calc_iou_from_points(prediction[0], prediction[1], prediction[2], prediction[3], sub[0], sub[1], sub[2], sub[3]) > iou_threshold:
                        # elimate box with lower confidence that has high iou with another box
                        predictions[i+j+1] = torch.zeros((field_length))

        # elimate empty rows
        predictions = predictions[predictions.any(dim=1)]
        
        return predictions
        

        