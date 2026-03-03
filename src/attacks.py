# src/attacks.py

import torch
import torch.nn.functional as F


def fgsm_random_start(model, images, labels, epsilon=8/255, alpha=8/255):

    device = images.device

    # Random Start
    random_noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = images + random_noise
    adv_images = torch.clamp(adv_images, 0, 1)

    adv_images.requires_grad = True

    # Forward pass
    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels)

    # Backward
    model.zero_grad()
    loss.backward()

    # FGSM Step
    grad = adv_images.grad.sign()
    adv_images = adv_images + alpha * grad

    # Project back into epsilon-ball
    eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
    adv_images = torch.clamp(images + eta, 0, 1).detach()

    return adv_images