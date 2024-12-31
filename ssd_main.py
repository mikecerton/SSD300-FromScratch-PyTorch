import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from ssd_head import SSD_Head
from ssd_backbone import ssd_backbone

class SSD_Main(nn.Module):
    def __init__(self, n_class = 91, n_bbox = None):
        super(SSD_Main, self).__init__()

        self.n_class = int(n_class)
        self.n_bbox = n_bbox


        self.backbone = ssd_backbone()
        self.head = SSD_Head(self.n_class, self.n_bbox)
        
    def forward(self, img_tensor):
        conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out = self.backbone(img_tensor)
        class_score, loc_bbox = self.head(conv4_3_out, conv7_out, conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out)
        return class_score, loc_bbox
    
    def load_all_weight(self):
        self.backbone.load_backbone_weight()
        self.head.cls_head.load_weight_ClsHead()
        self.head.loc_head.load_weight_LocHead()


def load_and_preprocess_image(image_path, output_size=(300, 300)):
    """
    Reads an image from the given path and preprocesses it to a PyTorch tensor with shape (1, 3, H, W).
    
    Parameters:
    - image_path (str): Path to the image file.
    - output_size (tuple): Desired output size as (height, width). Default is (300, 300).

    Returns:
    - torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W).
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(output_size),  # Resize to the specified size
        transforms.ToTensor(),           # Convert image to tensor with shape (C, H, W)
    ])

    # Apply the transformation
    image_tensor = transform(image)

    # Add a batch dimension to get shape (1, 3, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

if __name__ == "__main__":

    random_image = torch.rand(1, 3, 300, 300)

    n_class = 91
    n_bbox = {
        'conv4_3': 4,
        'conv7': 6,
        'conv8_2': 6,
        'conv9_2': 6,
        'conv10_2': 4,
        'conv11_2': 4
    }
    
    ssd =  SSD_Main(n_class, n_bbox)

    image_tensor = load_and_preprocess_image("D:\SSD300-FromScratch-PyTorch\download.jpg")

    class_score, loc_bbox = ssd(image_tensor)

    """Print output shapes"""
    print("class_score")
    for i, out in enumerate(class_score):
        print(f"Output {i} shape: {out.shape}")

    """Print output shapes"""
    print("loc_bbox")
    for i, out in enumerate(loc_bbox):
        print(f"Output {i} shape: {out.shape}")

    