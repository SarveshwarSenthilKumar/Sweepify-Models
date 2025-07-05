from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(img: Image.Image):
    # Resize and convert to tensor (dummy, adjust as needed for real model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension 