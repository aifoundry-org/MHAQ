import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from src.models.resnet.resnet_cifar import resnet20_cifar10


# -----------------------------
# 2. Learning Rate Scheduler
# -----------------------------
class CustomLRScheduler(_LRScheduler):
    """
    Imitates the Keras lr_schedule() function:
        - LR = 1e-3 by default
        - If epoch > 180: LR *= 0.5e-3
        - If epoch > 160: LR *= 1e-3
        - If epoch > 120: LR *= 1e-2
        - If epoch > 80:  LR *= 1e-1
    """
    def __init__(self, optimizer, last_epoch=-1):
        self.initial_lr = 5e-4
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lr = self.initial_lr
        if epoch < 10:
            lr = 0
        elif epoch > 80:
            lr *= 0.5e-3
        elif epoch > 60:
            lr *= 1e-3
        elif epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return [lr for _ in self.optimizer.param_groups]

# -----------------------------
# 6. Training & Evaluation
# -----------------------------
def train_one_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

# -----------------------------
# 7. Putting It All Together
# -----------------------------
def main():
    # Hyperparameters and settings
    batch_size = 32
    epochs = 100
    data_augmentation = True
    normalize = True
    num_classes = 10

    # For a ResNet20, depth=6*3+2=20
    n = 3
    depth = n * 6 + 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Data Preparation
    # -----------------------------
    # We will replicate your data augmentation:
    #  - Random horizontal flip
    #  - Random shift (via RandomCrop)
    #  - Normalize or subtract pixel mean as requested

    # 1. Load the raw training / test sets without transform
    train_set_raw = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )
    test_set_raw = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True
    )

    # 2. Compute pixel mean if requested
    if normalize:
        # Compute the mean over the entire training set
        #data_array = train_set_raw.data  # shape (50000, 32, 32, 3)
        #mean = data_array.mean(axis=(0,1,2)) / 255.0
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.247, 0.243, 0.261]
    else:
        mean = np.zeros(3)
        std = np.ones(3)

    # 3. Build transforms
    if data_augmentation:
        # roughly replicate Keras shift & flip
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),  # Subtract mean
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)
        ])
        print("Using real-time data augmentation.")
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        print("Not using data augmentation.")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 4. Create actual datasets with transforms
    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=test_transform
    )

    # 5. DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = resnet20_cifar10(pretrained=True).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()

    # Weight decay of 1e-4 as in Keras l2(1e-4)
    optimizer = optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler that mimics your Keras code
    scheduler = CustomLRScheduler(optimizer)

    # For saving best model
    save_dir = os.path.join(os.getcwd(), "saved_models_pytorch")
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, f"cifar10_ResNet{depth}v1_best.th")
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    best_loss, best_acc = evaluate(model, device, test_loader, criterion)
    print(f"Best Model Test Loss: {best_loss:.4f} | Best Model Test Acc: {best_acc:.2f}%")
    #best_acc = 92.62


    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(epochs):
        # Update LR from custom scheduler
        scheduler.step(epoch)

        print(f"\nEpoch [{epoch+1}/{epochs}]  |  Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        # Checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'state_dict': model.state_dict()}, best_model_path)
            print(f"Model saved to {best_model_path} (Accuracy: {best_acc:.2f}%)")
        elif test_acc < best_acc:
            model.load_state_dict(torch.load(best_model_path)['state_dict'])

    # -----------------------------
    # Final Evaluation
    # -----------------------------
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    final_loss, final_acc = evaluate(model, device, test_loader, criterion)
    print(f"Best Model Test Loss: {final_loss:.4f} | Best Model Test Acc: {final_acc:.2f}%")

if __name__ == "__main__":
    main()
