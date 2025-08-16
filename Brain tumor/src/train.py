import torch
import torch.nn as nn
import torch.optim as optim
from src.model import BrainTumorCNN
from src.dataset import get_dataloaders
from src.utils import accuracy


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"
    train_loader, val_loader, num_classes = get_dataloaders(data_dir)

    model = BrainTumorCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):  # small for demo
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "outputs/model.pth")

if __name__ == "__main__":
    train()
