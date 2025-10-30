import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from .model import get_transforms

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def preprocess_image(image, transforms):
    return transforms(image).unsqueeze(0)

def predict(model, input_tensor, device='cpu'):
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence

def display_image_with_heatmap(original_image, heatmap, title="Fracture Detection"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Original X-ray")
    ax[0].axis('off')
    ax[1].imshow(heatmap)
    ax[1].set_title(title)
    ax[1].axis('off')
    plt.show()

def save_heatmap(heatmap, save_path):
    plt.imshow(heatmap)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
