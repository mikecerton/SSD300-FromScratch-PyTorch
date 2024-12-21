

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision.transforms as T
from torchvision.ops import nms
from collections import OrderedDict

from PIL import Image

class Backbone_VGG16(nn.Module):
    def __init__(self):
        super(Backbone_VGG16, self).__init__()

        self.backbone_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True))
        
        self.backbone_2 = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
                    nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)),
                    nn.ReLU(inplace=True)
                )
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(inplace=True)
            )
        ])


    def forward(self, x):
        x = self.backbone_1(x)
        return x
    
    def load_weight_backbone_1(self):

        weights = SSD300_VGG16_Weights.DEFAULT
        pretrained_model = ssd300_vgg16(weights=weights)
        
        li = list(pretrained_model.backbone.children())
        
        self.backbone_1.load_state_dict(li[0].state_dict())
        

if __name__ == "__main__":

    random_image = torch.rand(1, 3, 300, 300)





    bb = Backbone_VGG16()

    bb.load_weight_backbone_1()
 
    b1 = bb.backbone_1(random_image)

    print(b1.shape)





    weights = SSD300_VGG16_Weights.DEFAULT
    pretrained_model = ssd300_vgg16(weights=weights)

    li = list(pretrained_model.backbone.children())

    pre1 = li[0](random_image)

    print(pre1.shape)


    print(torch.equal(pre1, b1))