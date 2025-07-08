"""
Training script for MNIST digit recognition.
Uses PyTorch with data augmentation and learning rate scheduling.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DigitRecognitionCNN, get_device, save_model


def get_data_loaders(batch_size=64):
 """
 Create train and test dataloaders with augmentation.

 Args:
 batch_size: Batch size for training

 Returns:
 Tuple of (train_loader, test_loader)
 """
 # Training transforms with augmentation
 train_transform = transforms.Compose(
 [
 transforms.RandomRotation(10),
 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
 transforms.ToTensor(),
 transforms.Normalize((0.1307,), (0.3081,)),
 ]
 )

 # Test transforms without augmentation
 test_transform = transforms.Compose(
 [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
 )

 # Load datasets
 train_dataset = datasets.MNIST(
 root="./data", train=True, download=True, transform=train_transform
 )

 test_dataset = datasets.MNIST(
 root="./data", train=False, download=True, transform=test_transform
 )

 # Create dataloaders
 train_loader = DataLoader(
 train_dataset,
 batch_size=batch_size,
 shuffle=True,
 num_workers=2,
 pin_memory=True,
 )

 test_loader = DataLoader(
 test_dataset,
 batch_size=batch_size,
 shuffle=False,
 num_workers=2,
 pin_memory=True,
 )

 return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
 """Train for one epoch."""
 model.train()
 running_loss = 0.0
 correct = 0
 total = 0

 for batch_idx, (data, target) in enumerate(train_loader):
 data, target = data.to(device), target.to(device)

 optimizer.zero_grad()
 output = model(data)
 loss = criterion(output, target)
 loss.backward()
 optimizer.step()

 running_loss += loss.item()
 _, predicted = output.max(1)
 total += target.size(0)
 correct += predicted.eq(target).sum().item()

 if (batch_idx + 1) % 100 == 0:
 print(
 f" Batch [{batch_idx + 1}/{len(train_loader)}] | "
 f"Loss: {running_loss / (batch_idx + 1):.4f} | "
 f"Acc: {100.0 * correct / total:.2f}%"
 )

 epoch_loss = running_loss / len(train_loader)
 epoch_acc = 100.0 * correct / total

 return epoch_loss, epoch_acc


def test_model(model, device, test_loader, criterion):
 """Evaluate model on test set."""
 model.eval()
 test_loss = 0
 correct = 0
 total = 0

 with torch.no_grad():
 for data, target in test_loader:
 data, target = data.to(device), target.to(device)
 output = model(data)
 test_loss += criterion(output, target).item()
 _, predicted = output.max(1)
 total += target.size(0)
 correct += predicted.eq(target).sum().item()

 test_loss /= len(test_loader)
 test_acc = 100.0 * correct / total

 return test_loss, test_acc


def train_digit_recognizer(epochs=20, batch_size=64, learning_rate=0.001):
 """
 Main training function for digit recognition model.

 Args:
 epochs: Number of training epochs
 batch_size: Training batch size
 learning_rate: Initial learning rate
 """
 # Setup
 os.makedirs("models", exist_ok=True)
 device = get_device()

 print("=" * 70)
 print("MNIST Digit Recognition Training")
 print("=" * 70)
 print(f"Device: {device}")

 # Load data
 print("\nLoading MNIST dataset...")
 train_loader, test_loader = get_data_loaders(batch_size)
 print(f"Training batches: {len(train_loader)}")
 print(f"Test batches: {len(test_loader)}")

 # Build model
 print("\nBuilding model...")
 model = DigitRecognitionCNN().to(device)

 # Loss and optimizer
 criterion = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 scheduler = optim.lr_scheduler.ReduceLROnPlateau(
 optimizer, mode="min", factor=0.5, patience=3, verbose=True
 )

 # Training loop
 print("\n" + "=" * 70)
 print("Starting training...")
 print("=" * 70 + "\n")

 best_acc = 0.0

 for epoch in range(epochs):
 print(f"Epoch [{epoch + 1}/{epochs}]")

 train_loss, train_acc = train_epoch(
 model, device, train_loader, optimizer, criterion, epoch
 )

 test_loss, test_acc = test_model(model, device, test_loader, criterion)

 scheduler.step(test_loss)

 print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
 print(f" Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

 # Save best model
 if test_acc > best_acc:
 best_acc = test_acc
 save_model(model, "models/digit_recognizer_best.pth")
 print(f" >>> New best model saved! (Acc: {best_acc:.2f}%)")

 print()

 # Save final model
 save_model(model, "models/digit_recognizer_final.pth")

 print("=" * 70)
 print(f"Training complete! Best test accuracy: {best_acc:.2f}%")
 print("=" * 70)

 return model


if __name__ == "__main__":
 train_digit_recognizer(epochs=20, batch_size=64, learning_rate=0.001)
