import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

class FractureDetectionModel(nn.Module):
    def __init__(self, num_classes=2, fine_tune=True):
        super().__init__()

        # ✅ Load pretrained ResNet-50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # 🔒 Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 🔓 Unfreeze last block if fine-tuning
        if fine_tune:
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

        # ✅ Replace classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def load_model(model_path=None, device="cpu"):
    model = FractureDetectionModel()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

