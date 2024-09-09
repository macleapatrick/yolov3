import torch
import torch.nn as nn
import os

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvModule, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class ResModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(ResModule, self).__init__()

        self.conv1 = ConvModule(in_channels, mid_channels, 1, stride, 0)
        self.conv2 = ConvModule(mid_channels, out_channels, 3, stride, 1)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        
        return x
    
class ConvModuleRepeater(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_repeats):
        super(ConvModuleRepeater, self).__init__()
        
        self.conv_module_repeater = nn.Sequential(*[ConvModule(in_channels, out_channels, kernel_size, 1, 1) for _ in range(num_repeats)])

    def forward(self, x):
        x = self.conv_module_repeater(x)
        
        return x
    
class ResModuleRepeater(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_repeats):
        super(ResModuleRepeater, self).__init__()
        
        self.res_module_repeater = nn.Sequential(*[ResModule(in_channels, mid_channels, out_channels) for _ in range(num_repeats)])
    
    def forward(self, x):
        x = self.res_module_repeater(x)
        
        return x

class Darknet53(nn.Module):
    def __init__(self, num_classes=1000, weights_path=None, classifier_only=True, load_weights=True):
        super(Darknet53, self).__init__()
        
        self.conv1 = ConvModule(3, 32, 3, 1, 1) #416x416
        self.conv2 = ConvModule(32, 64, 3, 2, 1) #208x208
        self.res1 = ResModuleRepeater(64, 32, 64, 1) #208x208
        self.conv3 = ConvModule(64, 128, 3, 2, 1) #104x104
        self.res2 = ResModuleRepeater(128, 64, 128, 2) #104x104
        self.conv4 = ConvModule(128, 256, 3, 2, 1) #52x52
        self.res3 = ResModuleRepeater(256, 128, 256, 8) #52x52
        self.conv5 = ConvModule(256, 512, 3, 2, 1) #26x26
        self.res4 = ResModuleRepeater(512, 256, 512, 8) #26x26
        self.conv6 = ConvModule(512, 1024, 3, 2, 1) #13x13
        self.res5 = ResModuleRepeater(1024, 512, 1024, 4) #13x13
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  #1024
        self.fc1 = nn.Linear(1024, num_classes)    
        
        self.weights_path = weights_path

        if weights_path and load_weights: 
            self.load_weights(weights_path)

        self.classifier_only = classifier_only

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        x = self.res3(x)
        
        if not self.classifier_only:
            split1 = x.clone()

        x = self.conv5(x)
        x = self.res4(x)
        
        if not self.classifier_only:
            split2 = x.clone()
        
        x = self.conv6(x)
        split3 = self.res5(x)
        
        if self.classifier_only:
            x = split3
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
        
        if not self.classifier_only:
            return split1, split2, split3
        else:
            return x
    
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