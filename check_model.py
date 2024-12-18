import torch
import torch.nn as nn
from my_SSD import my_SSD
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights


def compare_state_dicts(model1, model2):

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for (k1, v1), (k2, v2) in zip(state_dict1.items(), state_dict2.items()):
        # print(f"{k1}   +++++   {v1.shape}    {k2}   +++++   {v2.shape}")
        print(f"   +++++   {v1.shape}       +++++   {v2.shape}")

def display_state(model, not_list = []):
    layer_details = []
    for key, tensor in model.state_dict().items():
        layer_name, param_type = key.rsplit('.', 1)
        size = tensor.size()
        layer = model.get_submodule(layer_name)  # Get the actual layer
        layer_details.append((layer_name, type(layer).__name__, param_type, size))

    for z in not_list:
        layer_details.pop(z)

    # Display the results
    print(f"{'Layer':<15} {'Type':<15} {'Param':<10} {'Shape':<20}")
    print("-" * 60)
    for layer_name, layer_type, param_type, size in layer_details:
        print(f"{layer_name:<15} {layer_type:<15} {param_type:<10} {str(size):<20}")





model_2 = my_SSD()
model_1 = pretrained_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

# compare_state_dicts(model_1, model_2)

not_list = [0]

display_state(model_2)
display_state(model_1, not_list)