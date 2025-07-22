"""
Training script for MNIST digit recognition.
Uses PyTorch with simple CNN architecture.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import ImageClassifier, get_device, save_model


def get_data_loaders(batch_size=32):
    """
    Create train and test dataloaders.

    Args:
        batch_size: Batch size for training

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Download MNIST dataset
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    # Wrap datasets in DataLoader for batching
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, loss_fn, epoch, epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        yhat = model(X)
        loss = loss_fn(yhat, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")
    return avg_loss


def test_model(model, device, test_loader):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            yhat = model(X)
            preds = torch.argmax(yhat, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def train_digit_recognizer(epochs=10, batch_size=32, learning_rate=0.001):
    """
    Main training function for digit recognition model.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    # Setup
    device = get_device()
    print("=" * 70)
    print("MNIST Digit Recognition Training")
    print("=" * 70)
    print(f"Device: {device}\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Build model
    print("Building model...")
    model = ImageClassifier().to(device)
    print(model)
    print()

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    for epoch in range(epochs):
        train_epoch(model, device, train_loader, optimizer, loss_fn, epoch, epochs)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    accuracy = test_model(model, device, test_loader)

    # Save model
    save_model(model, "models/mnist_cnn_model.pt")

    print("\n" + "=" * 70)
    print(f"Training complete! Final accuracy: {accuracy * 100:.2f}%")
    print("=" * 70)

    return model


if __name__ == "__main__":
    train_digit_recognizer(epochs=10, batch_size=32, learning_rate=0.001)
