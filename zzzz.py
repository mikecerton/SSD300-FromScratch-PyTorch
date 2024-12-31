import torch
from torchvision import transforms
from PIL import Image


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


image_tensor = load_and_preprocess_image("D:\SSD300-FromScratch-PyTorch\download.jpg")
print("Image tensor shape:", image_tensor.shape)
