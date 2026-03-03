# src/trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attacks import fgsm_random_start


def train_hfat(student, teacher, train_loader, optimizer, device, alpha=0.7, temperature=4):

    student.train()
    teacher.eval()

    total_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images
        adv_images = fgsm_random_start(student, images, labels)

        optimizer.zero_grad()

        # Student predictions
        student_adv = student(adv_images)
        student_clean = student(images)

        # Teacher prediction (clean only)
        with torch.no_grad():
            teacher_clean = teacher(images)

        # Cross Entropy Loss (robustness)
        ce_loss = F.cross_entropy(student_adv, labels)

        # Knowledge Distillation Loss
        kd_loss = F.kl_div(
            F.log_softmax(student_clean / temperature, dim=1),
            F.softmax(teacher_clean / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hybrid Loss
        loss = alpha * ce_loss + (1 - alpha) * kd_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)