# train_teacher.py

import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import get_cifar10_loaders
from src.models import SmallCNN


def train_teacher(epochs=10, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load CIFAR-10
    train_loader, test_loader = get_cifar10_loaders()

    # Teacher model
    model = SmallCNN().to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

        # Evaluate after each epoch
        acc = evaluate(model, test_loader, device)

        # Save best teacher
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/teacher.pth")
            print("✅ Teacher checkpoint saved!")

    print("Training complete.")
    print("Best Test Accuracy:", best_acc)


def evaluate(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    return acc


if __name__ == "__main__":
    train_teacher(epochs=15)
