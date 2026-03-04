import os
import torch
import torch.nn as nn
import torch.optim as optim

# Import existing project modules
from src.data_loader import get_cifar10_loaders
from src.models import SmallCNN
from src.trainer import train_hfat
# Changed 'evaluate_adversarial' to 'evaluate_robust' to match src/evaluator.py
from src.evaluator import evaluate_clean, evaluate_robust

def run_full_pipeline():
    # 1. Setup Device and Directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        print("Created 'checkpoints' directory.")

    # 2. Load Data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()

    # ---------------------------------------------------------
    # STEP 1: TRAIN TEACHER MODEL
    # ---------------------------------------------------------
    print("\n--- Phase 1: Training Teacher Model ---")
    teacher = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_t = optim.Adam(teacher.parameters(), lr=0.001)
    
    epochs_teacher = 10
    best_acc = 0
    for epoch in range(epochs_teacher):
        teacher.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_t.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_t.step()
            total_loss += loss.item()
        
        # Evaluate teacher
        acc = evaluate_clean(teacher, test_loader, device)
        print(f"Teacher Epoch [{epoch+1}/{epochs_teacher}] - Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(teacher.state_dict(), "checkpoints/teacher.pth")
    
    print(f"Teacher training complete. Best Acc: {best_acc:.2f}%. Saved to checkpoints/teacher.pth")

    # ---------------------------------------------------------
    # STEP 2: TRAIN STUDENT MODEL (HFAT)
    # ---------------------------------------------------------
    print("\n--- Phase 2: Training Student Model (HFAT) ---")
    student = SmallCNN().to(device)
    optimizer_s = optim.Adam(student.parameters(), lr=0.001)
    
    # Reload best teacher for distillation
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    epochs_student = 10
    for epoch in range(epochs_student):
        # Use the HFAT training function from src/trainer.py
        train_loss = train_hfat(student, teacher, train_loader, optimizer_s, device)
        
        # Evaluate student
        clean_acc = evaluate_clean(student, test_loader, device)
        print(f"Student Epoch [{epoch+1}/{epochs_student}] - Loss: {train_loss:.4f}, Clean Acc: {clean_acc:.2f}%")
    
    torch.save(student.state_dict(), "checkpoints/hfat_student.pth")
    print("Student training complete. Saved to checkpoints/hfat_student.pth")

    # ---------------------------------------------------------
    # STEP 3: FINAL EVALUATION (ROBUSTNESS)
    # ---------------------------------------------------------
    print("\n--- Phase 3: Final Robustness Evaluation (PGD-20) ---")
    # Updated to call evaluate_robust instead of evaluate_adversarial
    robust_acc = evaluate_robust(student, test_loader, device)
    print(f"Final Student Robust Accuracy (PGD-20): {robust_acc:.2f}%")

if __name__ == "__main__":
    run_full_pipeline()