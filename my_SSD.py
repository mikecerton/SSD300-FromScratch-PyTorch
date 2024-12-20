

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision.transforms as T
from torchvision.ops import nms

from PIL import Image

class Backbone_VGG16(nn.Module):
    def __init__(self):
        super(Backbone_VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))  # (N, 64, 300, 300)
        x = F.relu(self.conv1_2(x))  # (N, 64, 300, 300)
        x = self.pool1(x)            # (N, 64, 150, 150)

        x = F.relu(self.conv2_1(x))  # (N, 128, 150, 150)
        x = F.relu(self.conv2_2(x))  # (N, 128, 150, 150)
        x = self.pool2(x)            # (N, 128, 75, 75)

        x = F.relu(self.conv3_1(x))  # (N, 256, 75, 75)
        x = F.relu(self.conv3_2(x))  # (N, 256, 75, 75)
        x = F.relu(self.conv3_3(x))  # (N, 256, 75, 75)
        x = self.pool3(x)            # (N, 256, 38, 38)

        x = F.relu(self.conv4_1(x))  # (N, 512, 38, 38)
        x = F.relu(self.conv4_2(x))  # (N, 512, 38, 38)
        x = F.relu(self.conv4_3(x))  # (N, 512, 38, 38)
        conv4_3_out = x              # (N, 512, 38, 38) ; output of conv4_3 for SSD head
        x = self.pool4(x)            # (N, 512, 19, 19)

        x = F.relu(self.conv5_1(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_2(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_3(x))  # (N, 512, 19, 19)
        x = self.pool5(x)            # (N, 512, 19, 19)
        
        return conv4_3_out, x
    
class Feature_extraction(nn.Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) 

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        

    def forward(self, x):

        x = F.relu(self.conv6(x))               # (N, 1024, 19, 19)

        x = F.relu(self.conv7(x))               # (N, 1024, 19, 19)
        conv7_out = x

        x = F.relu(self.conv8_1(x))             # (N, 256, 19, 19)
        x = F.relu(self.conv8_2(x))             # (N, 512, 10, 10)
        conv8_2_out = x                         # (N, 512, 10, 10) ; output of conv8_2 for SSD head

        x = F.relu(self.conv9_1(x))             # (N, 128, 10, 10)
        x = F.relu(self.conv9_2(x))             # (N, 256, 5, 5)
        conv9_2_out = x                         # (N, 256, 5, 5) ; output of conv9_2 for SSD head

        x = F.relu(self.conv10_1(x))            # (N, 128, 5, 5)
        x = F.relu(self.conv10_2(x))            # (N, 256, 3, 3)
        conv10_2_out = x                        # (N, 256, 3, 3) ; output of conv10_2 for SSD head

        x = F.relu(self.conv11_1(x))            # (N, 128, 3, 3)
        conv11_2_out = F.relu(self.conv11_2(x)) # (N, 256, 1, 1) ; output of conv11_2 for SSD head

        return conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
    
class SSD_head(nn.Module):
    def __init__(self, n_class):
        super(SSD_head, self).__init__()
        self.n_class = n_class

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cls_conv4_3 = nn.Conv2d(512, 4 * n_class, kernel_size=3, padding=1)
        self.cls_conv7 = nn.Conv2d(1024, 6 * n_class, kernel_size=3, padding=1)
        self.cls_conv8_2 = nn.Conv2d(512, 6 * n_class, kernel_size=3, padding=1)
        self.cls_conv9_2 = nn.Conv2d(256, 6 * n_class, kernel_size=3, padding=1)
        self.cls_conv10_2 = nn.Conv2d(256, 4 * n_class, kernel_size=3, padding=1)
        self.cls_conv11_2 = nn.Conv2d(256, 4 * n_class, kernel_size=3, padding=1)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):

        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)


        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
   
        # Predict classes in localization boxes
        c_conv4_3 = self.cls_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        
        c_conv7 = self.cls_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
      
        c_conv8_2 = self.cls_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
       
        c_conv9_2 = self.cls_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
       
        c_conv10_2 = self.cls_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
       
        c_conv11_2 = self.cls_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
       
        return l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2, c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2

class my_SSD(nn.Module):
    def __init__(self, n_class=20):
        super(my_SSD, self).__init__()
        self.backbone = Backbone_VGG16()
        self.extraction = Feature_extraction()
        self.head = SSD_head(n_class)
        self.n_class = n_class

        # Placeholder for prior boxes, typically predefined
        self.priors = torch.rand(8732, 4)  # Example: 8732 prior boxes

    def forward(self, x):
        conv4_3_out, x = self.backbone(x)
        conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = self.extraction(x)
        l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2, c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2 = self.head(
            conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
        )

        locs = torch.cat([l.view(l.size(0), -1, 4) for l in [l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2]], dim=1)
        confs = torch.cat([c.view(c.size(0), -1, self.n_class) for c in [c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2]], dim=1)

        return locs, confs

def get_backbone_weight():
    from torchvision.models.detection import ssd300_vgg16
    from torchvision.models.detection.ssd import SSD300_VGG16_Weights

    weights = SSD300_VGG16_Weights.DEFAULT
    pretrained_model = ssd300_vgg16(weights=weights)

    li = list(pretrained_model.backbone.children())

    a = li[0]
    b = li[1][0][0:7]
    c = li[1][0][7][0]

    la_list = list(a) + list(b) + [c]

    combined_seq = nn.Sequential(*la_list)

    return combined_seq

if __name__ == "__main__":
    bb = Backbone_VGG16()

    # Get the pre-trained backbone weights
    pre_model = get_backbone_weight()

    # Load the pre-trained backbone weights into your custom model
    bb.load_state_dict(pre_model.state_dict(), strict=False)  # strict=False if the architecture differs slightly

    # Print the model to verify it's loaded
    # print(bb)

    # Example: Forward pass with random image
    random_image = torch.rand(1, 3, 300, 300)  # Example input (batch size of 1, 3 channels, 300x300 image)
    output = bb(random_image)

    output_2 = pre_model(random_image)

    print(torch.equal(output[0], output_2[0]))



    
