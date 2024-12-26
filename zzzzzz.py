
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision
from torch import nn
import torch


weights = SSD300_VGG16_Weights.DEFAULT
pretrained_model = ssd300_vgg16(weights=weights)

# print(pretrained_model)

# for a in pretrained_model.head.children():
#     print(a)

for a in pretrained_model.backbone.children():
    print(a)

# print(pretrained_model)

# li = list(pretrained_model.backbone.children())

# # print(li[0])          # pass

# print(li[1])           # ghost
# # print(li[1][0][0:7])  # pass
# # print(li[1][0][7][0])    # pass

# # print(li[1][1])       # pass

# # print(li[1][2])         # pass       

# # print(li[1][3])         # pass

# # print(li[1][4])         # pass

# a = li[0]
# b = li[1][0][0:7]
# c = li[1][0][7][0]

# # Combine layers into a single list
# la_list = list(a) + list(b) + [c]

# # Create the combined Sequential model
# combined_seq = nn.Sequential(*la_list)

# # Print the result
# print(combined_seq)


def compare_modulelists(list1, list2):
    if len(list1) != len(list2):
        return False
    for m1, m2 in zip(list1, list2):
        if not compare_sequential(m1, m2):
            return False
    return True

def compare_sequential(seq1, seq2):
    # Compare the sequence of layers in Sequential, ignoring the parameters
    if len(seq1) != len(seq2):
        return False
    for layer1, layer2 in zip(seq1, seq2):
        if type(layer1) != type(layer2):
            return False
    return True

def compare_state_dicts(state_dict1, state_dict2):
    # Check if the state_dicts have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False
    
    # Compare the actual parameters (weights and biases)
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    
    return True






