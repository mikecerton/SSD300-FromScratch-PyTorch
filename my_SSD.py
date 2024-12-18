
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

class backbone_VGG16(nn.Module):
    def __init__(self):
        super(backbone_VGG16, self).__init__()

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
        conv4_3_out = x              # (N, 512, 38, 38) ; output of conv4_3 for detection head
        x = self.pool4(x)            # (N, 512, 19, 19)

        x = F.relu(self.conv5_1(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_2(x))  # (N, 512, 19, 19)
        x = F.relu(self.conv5_3(x))  # (N, 512, 19, 19)
        x = self.pool5(x)            # (N, 512, 19, 19)
        
        return conv4_3_out, x
    
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

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
        x = F.relu(self.conv8_1(x))             # (N, 256, 19, 19)
        x = F.relu(self.conv8_2(x))             # (N, 512, 10, 10)
        conv8_2_out = x                         # (N, 512, 10, 10) ; output of conv8_2 for detection head

        x = F.relu(self.conv9_1(x))             # (N, 128, 10, 10)
        x = F.relu(self.conv9_2(x))             # (N, 256, 5, 5)
        conv9_2_out = x                         # (N, 256, 5, 5) ; output of conv9_2 for detection head

        x = F.relu(self.conv10_1(x))            # (N, 128, 5, 5)
        x = F.relu(self.conv10_2(x))            # (N, 256, 3, 3)
        conv10_2_out = x                        # (N, 256, 3, 3) ; output of conv10_2 for detection head

        x = F.relu(self.conv11_1(x))            # (N, 128, 3, 3)
        conv11_2_out = F.relu(self.conv11_2(x)) # (N, 256, 1, 1) ; output of conv11_2 for detection head

        return conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
    

class my_SSD(nn.Module):
    def __init__(self):
        super(my_SSD, self).__init__()

        self.backbone = backbone_VGG16()
        self.extraction = feature_extraction()

    def forward(self, x):
        conv4_3_out, x = self.backbone(x)
        conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = self.extraction(x)
        return conv4_3_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
    
    def load_pretrained_weights(self, weights=SSD300_VGG16_Weights.DEFAULT):
        
        pretrained_model = ssd300_vgg16(weights=weights)
        pretrained_state_dict = pretrained_model.state_dict()

        model_state_dict = self.state_dict()

        filtered_state_dict = {
            k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        model_state_dict.update(filtered_state_dict)
        self.load_state_dict(model_state_dict)

        print(f"Loaded {len(filtered_state_dict)} layers from pretrained weights.")
    

if __name__ == "__main__":

    model_1 = my_SSD()
    model_1.load_pretrained_weights()

    print(model_1)
