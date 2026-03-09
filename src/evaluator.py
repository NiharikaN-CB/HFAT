import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import get_cifar10_loaders
from src.models import SmallCNN


# -------------------------------
# PGD ATTACK
# -------------------------------
def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, iters=20):

    images = images.clone().detach()
    adv_images = images.clone().detach()

    # random start
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(iters):

        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = adv_images.grad.sign()

        adv_images = adv_images + alpha * grad

        # projection
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)

        adv_images = torch.clamp(images + eta, 0, 1).detach()

    return adv_images


# -------------------------------
# CLEAN ACCURACY
# -------------------------------
def evaluate_clean(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


# -------------------------------
# ROBUST ACCURACY (PGD)
# -------------------------------
def evaluate_robust(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        adv_images = pgd_attack(model, images, labels)

        outputs = model(adv_images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    return acc


# -------------------------------
# MAIN EVALUATION FUNCTION
# -------------------------------
def run_evaluation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    # Load teacher
    teacher = SmallCNN().to(device)
    teacher.load_state_dict(torch.load("checkpoints/teacher.pth"))
    teacher.eval()

    # Load HFAT student
    student = SmallCNN().to(device)
    student.load_state_dict(torch.load("checkpoints/hfat_student.pth"))
    student.eval()

    print("\nEvaluating Teacher Model...")
    teacher_clean = evaluate_clean(teacher, test_loader, device)
    teacher_robust = evaluate_robust(teacher, test_loader, device)

    print("\nEvaluating HFAT Model...")
    student_clean = evaluate_clean(student, test_loader, device)
    student_robust = evaluate_robust(student, test_loader, device)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(f"Teacher Clean Accuracy: {teacher_clean:.2f}%")
    print(f"Teacher Robust Accuracy (PGD-20): {teacher_robust:.2f}%")

    print()

    print(f"HFAT Clean Accuracy: {student_clean:.2f}%")
    print(f"HFAT Robust Accuracy (PGD-20): {student_robust:.2f}%")

    # -------------------------------
    # RESULTS TABLE
    # -------------------------------
    results = {
        "Model": ["Teacher", "HFAT Student"],
        "Clean Accuracy (%)": [teacher_clean, student_clean],
        "Robust Accuracy (PGD-20) (%)": [teacher_robust, student_robust]
    }

    df = pd.DataFrame(results)

    print("\nResults Table:")
    print(df.to_string(index=False))

    # -------------------------------
    # GRAPH
    # -------------------------------
    labels = ["Teacher", "HFAT Student"]
    clean = [teacher_clean, student_clean]
    robust = [teacher_robust, student_robust]

    x = range(len(labels))

    plt.figure(figsize=(8,5))

    plt.bar(x, clean, width=0.4, label="Clean Accuracy")
    plt.bar([i + 0.4 for i in x], robust, width=0.4, label="Robust Accuracy (PGD-20)")

    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Teacher vs HFAT Student Performance")

    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_evaluation()