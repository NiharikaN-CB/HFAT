# train_student.py

import torch
import torch.optim as optim
from src.data_loader import get_cifar10_loaders
from src.models import SmallCNN
from src.trainer import train_hfat


def train_student(epochs=10, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    # Load teacher
    teacher = SmallCNN().to(device)
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    # Student
    student = SmallCNN().to(device)

    optimizer = optim.Adam(student.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(epochs):

        loss = train_hfat(student, teacher, train_loader, optimizer, device)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save(student.state_dict(), "checkpoints/hfat_student.pth")
            print("✅ HFAT student checkpoint saved!")

    print("HFAT training complete.")


if __name__ == "__main__":
    train_student(epochs=10)