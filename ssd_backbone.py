import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights


class ssd_backbone(nn.Module):
    def __init__(self):
        super(ssd_backbone, self).__init__()

        self.scale_weight = nn.Parameter(torch.ones(512) * 20)

        self.vgg_base = nn.Sequential(
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
        
        self.extra_feature = nn.ModuleList([
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

        outputs = []

        x = self.vgg_base(x)
        rescaled = self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)

        outputs.append(rescaled) 

        for layer in self.extra_feature:
            x = layer(x)
            outputs.append(x)

        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], 
    
    def load_backbone_weight(self):
        self.load_weight_backbone_1()
        self.load_weight_backbone_2()
    
    def load_weight_backbone_1(self):

        weights = SSD300_VGG16_Weights.DEFAULT
        pretrained_model = ssd300_vgg16(weights=weights)

        li = list(pretrained_model.backbone.children())
        
        self.vgg_base.load_state_dict(li[0].state_dict())
    
    def load_weight_backbone_2(self):

        weights = SSD300_VGG16_Weights.DEFAULT
        pretrained_model = ssd300_vgg16(weights=weights)

        li = list(pretrained_model.backbone.children())
        
        self.extra_feature.load_state_dict(li[1].state_dict())
    

if __name__ == "__main__":

    random_image = torch.rand(1, 3, 300, 300)

    bb = ssd_backbone()
    bb.load_weight_backbone_1()
    bb.load_weight_backbone_2()
    conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = bb(random_image)

    print("conv4_3_out shape:", conv4_3_out.shape)
    print("conv7_out shape:", conv7_out.shape)
    print("conv8_2_out shape:", conv8_2_out.shape)
    print("conv9_2_out shape:", conv9_2_out.shape)
    print("conv10_2_out shape:", conv10_2_out.shape)
    print("conv11_2_out shape:", conv11_2_out.shape)


    # weights = SSD300_VGG16_Weights.DEFAULT
    # pretrained_model = ssd300_vgg16(weights=weights)

    # conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = pretrained_model.backbone(random_image)


