import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from src.model import FractureDetectionModel, get_transforms

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Assuming BoneFractureDataset structure: training/fractured/ and training/not_fractured/
        fractured_dir = os.path.join(root_dir, 'fractured')
        not_fractured_dir = os.path.join(root_dir, 'not_fractured')

        # Load fractured images (label 1)
        if os.path.exists(fractured_dir):
            for img_file in os.listdir(fractured_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(fractured_dir, img_file))
                    self.labels.append(1)

        # Load not_fractured images (label 0)
        if os.path.exists(not_fractured_dir):
            for img_file in os.listdir(not_fractured_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(not_fractured_dir, img_file))
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    transform = get_transforms()
    train_dataset = BoneFractureDataset(os.path.join(data_dir, 'training'), transform=transform)
    val_dataset = BoneFractureDataset(os.path.join(data_dir, 'testing'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = FractureDetectionModel()
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'models/fracture_detection_model.pth')
    print("Model saved to models/fracture_detection_model.pth")

if __name__ == "__main__":
    # Assuming data is in data/BoneFractureDataset/
    train_model('data/BoneFractureDataset/')
