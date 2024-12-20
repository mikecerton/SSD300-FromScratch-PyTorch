from my_SSD import my_SSD
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision
from torch import nn


weights = SSD300_VGG16_Weights.DEFAULT
pretrained_model = ssd300_vgg16(weights=weights)

# print(pretrained_model)

# for a in pretrained_model.head.children():
#     print(a)

# for a in pretrained_model.backbone.children():
#     print(type(a))

li = list(pretrained_model.backbone.children())

# print(li[0])          # pass

# # print(li[1][0])       ghost
# print(li[1][0][0:7])  # pass
# print(li[1][0][7][0])    # pass

# print(li[1][1])       # pass

# print(li[1][2])         # pass       

# print(li[1][3])         # pass

# print(li[1][4])         # pass

a = li[0]
b = li[1][0][0:7]
c = li[1][0][7][0]

# Combine layers into a single list
la_list = list(a) + list(b) + [c]

# Create the combined Sequential model
combined_seq = nn.Sequential(*la_list)

# Print the result
print(combined_seq)
