import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights


# use reshape function in tensor as mmuch as possible

class SSD_Head(nn.Module):
    def __init__(self, n_class = 91, n_bbox = None):
        super(SSD_Head, self).__init__()


        self.n_class = n_class
        self.n_bbox = n_bbox

        self.cls_head = classi_head(self.n_class, self.n_bbox)
        self.loc_head = Loca_head(n_bbox)

    def forward(self, conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out):
        class_score = self.cls_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)
        loc_bbox = self.loc_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)
        return class_score, loc_bbox
    

class classi_head(nn.Module):
    def __init__(self, n_class=91, n_bbox=None):
        super(classi_head, self).__init__()

        self.n_class = n_class
        self.n_bbox = n_bbox

        self.cls_head = nn.ModuleList([
            nn.Conv2d(512, n_class * n_bbox['conv4_3'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, n_class * n_bbox['conv7'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, n_class * n_bbox['conv8_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, n_class * n_bbox['conv9_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, n_class * n_bbox['conv10_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, n_class * n_bbox['conv11_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ])

    def forward(self, conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out):
        conv_outs = [
            self.cls_head[0](conv4_3_out), # (N, 4 * n_classes, 38, 38)
            self.cls_head[1](conv7_out),   # (N, 6 * n_classes, 19, 19)
            self.cls_head[2](conv8_2_out), # (N, 6 * n_classes, 10, 10)
            self.cls_head[3](conv9_2_out), # (N, 6 * n_classes, 5, 5)
            self.cls_head[4](conv10_2_out),# (N, 4 * n_classes, 3, 3)
            self.cls_head[5](conv11_2_out) # (N, 4 * n_classes, 1, 1)
        ]

        batch_size = conv4_3_out.size(0)
        reshaped_outs = []

        for idx, out in enumerate(conv_outs):
            # Reshape to (N, H * W * num_boxes, n_classes)
            reshaped = out.permute(0, 2, 3, 1)  # (N, H, W, C)
            reshaped = reshaped.reshape(batch_size, -1, self.n_class)  # (N, H * W * num_boxes, n_classes)
            reshaped_outs.append(reshaped)

        return reshaped_outs
    
    def load_weight_ClsHead(self):
        weights = SSD300_VGG16_Weights.DEFAULT
        pretrained_model = ssd300_vgg16(weights=weights)
        
        li = list(pretrained_model.head.children())

        lli = list(li[0].children())

        self.cls_head.load_state_dict(lli[0].state_dict())

class Loca_head(nn.Module):
    def __init__(self, n_bbox=None):
        super(Loca_head, self).__init__()

        self.loc_head = nn.ModuleList([
            nn.Conv2d(512, 4 * n_bbox['conv4_3'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 4 * n_bbox['conv7'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 4 * n_bbox['conv8_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 4 * n_bbox['conv9_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 4 * n_bbox['conv10_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 4 * n_bbox['conv11_2'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ])

    def forward(self, conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out):
        conv_outs = [
            self.loc_head[0](conv4_3_out),  # (N, 16, 38, 38)
            self.loc_head[1](conv7_out),   # (N, 24, 19, 19)
            self.loc_head[2](conv8_2_out), # (N, 24, 10, 10)
            self.loc_head[3](conv9_2_out), # (N, 24, 5, 5)
            self.loc_head[4](conv10_2_out),# (N, 16, 3, 3)
            self.loc_head[5](conv11_2_out) # (N, 16, 1, 1)
        ]

        batch_size = conv4_3_out.size(0)
        reshaped_outs = []

        for idx, out in enumerate(conv_outs):
            # Reshape to (N, H * W * num_boxes, 4)
            num_boxes = 4  # Always 4 coordinates for bounding boxes (x, y, w, h)
            reshaped = out.permute(0, 2, 3, 1)  # (N, H, W, C)
            reshaped = reshaped.reshape(batch_size, -1, num_boxes)  # (N, H * W * num_boxes, 4)
            reshaped_outs.append(reshaped)

        return reshaped_outs
    
    def load_weight_LocHead(self):
        weights = SSD300_VGG16_Weights.DEFAULT
        pretrained_model = ssd300_vgg16(weights=weights)
        
        li = list(pretrained_model.head.children())

        lli = list(li[1].children())

        self.loc_head.load_state_dict(lli[0].state_dict())

def compare_state_dicts(state_dict1, state_dict2):
    # Check if the state_dicts have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False
    
    # Compare the actual parameters (weights and biases)
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    
    return True

if __name__ == "__main__":

    n_class = 91
    n_bbox = {
        'conv4_3': 4,
        'conv7': 6,
        'conv8_2': 6,
        'conv9_2': 6,
        'conv10_2': 4,
        'conv11_2': 4
    }
    

    """Simulate inputs"""
    conv4_3_out = torch.randn(1, 512, 38, 38)
    conv7_out = torch.randn(1, 1024, 19, 19)
    conv8_2_out = torch.randn(1, 512, 10, 10)
    conv9_2_out = torch.randn(1, 256, 5, 5)
    conv10_2_out = torch.randn(1, 256, 3, 3)
    conv11_2_out = torch.randn(1, 256, 1, 1)

    my_head = SSD_Head(n_class, n_bbox)
    my_head.load_head_weight()
    class_score, loc_bbox = my_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)

    """Print output shapes"""
    print("class_score")
    for i, out in enumerate(class_score):
        print(f"Output {i} shape: {out.shape}")

    """Print output shapes"""
    print("loc_bbox")
    for i, out in enumerate(loc_bbox):
        print(f"Output {i} shape: {out.shape}")





# ++++++++++++++++++++++++++++++++++ save space ++++++++++++++++++++++++++++++++++++++++++

    # """Simulate inputs"""
    # conv4_3_out = torch.randn(1, 512, 38, 38)
    # conv7_out = torch.randn(1, 1024, 19, 19)
    # conv8_2_out = torch.randn(1, 512, 10, 10)
    # conv9_2_out = torch.randn(1, 256, 5, 5)
    # conv10_2_out = torch.randn(1, 256, 3, 3)
    # conv11_2_out = torch.randn(1, 256, 1, 1)

    # my_head = SSD_Head(n_class, n_bbox)
    # class_score, loc_bbox = my_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)

    # """Print output shapes"""
    # print("class_score")
    # for i, out in enumerate(class_score):
    #     print(f"Output {i} shape: {out.shape}")

    # """Print output shapes"""
    # print("loc_bbox")
    # for i, out in enumerate(loc_bbox):
    #     print(f"Output {i} shape: {out.shape}")









    # my_class_head = classi_head(n_class, n_bbox)
    # outputs = my_class_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)

    # my_loca_head = Loca_head(n_bbox)
    # outputs = my_loca_head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)

    # """Print output shapes"""
    # for i, out in enumerate(outputs):
    #     print(f"Output {i} shape: {out.shape}")




        # my_class_head = classi_head(n_class, n_bbox)
    # my_class_head.load_weight_ClsHead()
    # model = my_class_head.cls_head

    # weights = SSD300_VGG16_Weights.DEFAULT
    # pretrained_model = ssd300_vgg16(weights=weights)
    # li = list(pretrained_model.head.children())
    # lli = list(li[0].children())
    
    # my_loca_head = Loca_head(n_bbox)
    # my_loca_head.load_weight_LocHead()
    # model = my_loca_head.loc_head

    # weights = SSD300_VGG16_Weights.DEFAULT
    # pretrained_model = ssd300_vgg16(weights=weights) 
    # li = list(pretrained_model.head.children())
    # lli = list(li[1].children())