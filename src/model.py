import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class FractureDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FractureDetectionModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def load_model(model_path=None):
    model = FractureDetectionModel()
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
