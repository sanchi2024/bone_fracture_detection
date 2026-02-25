import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from src.model import FractureDetectionModel, get_transforms

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        fractured_dir = os.path.join(root_dir, 'fractured')
        not_fractured_dir = os.path.join(root_dir, 'not_fractured')

        if os.path.exists(fractured_dir):
            for img in os.listdir(fractured_dir):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(fractured_dir, img))
                    self.labels.append(1)

        if os.path.exists(not_fractured_dir):
            for img in os.listdir(not_fractured_dir):
                if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(not_fractured_dir, img))
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(data_dir):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    EPOCHS = 3   # 👈 quick retraining for graph

    print(f"Using device: {DEVICE}")

    transform = get_transforms()

    train_dataset = BoneFractureDataset(os.path.join(data_dir, "training"), transform)
    val_dataset = BoneFractureDataset(os.path.join(data_dir, "testing"), transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FractureDetectionModel().to(DEVICE)

    model_path = "models/fracture_detection_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        print("✅ Loaded previous model weights")

    for param in model.resnet.parameters():
        param.requires_grad = False

    for param in model.resnet.layer4.parameters():
        param.requires_grad = True

    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.resnet.layer4.parameters(), "lr": 1e-4},
        {"params": model.resnet.fc.parameters(), "lr": 1e-3}
    ])

    train_acc_list = []
    val_acc_list = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        train_acc_list.append(train_acc)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 📊 Accuracy Graph
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, train_acc_list, label="Training Accuracy")
    plt.plot(epochs_range, val_acc_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.savefig("models/accuracy_graph.png")
    plt.show()

    print("✅ Accuracy graph saved to models/accuracy_graph.png")


if __name__ == "__main__":
    train_model("model")

