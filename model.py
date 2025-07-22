"""
PyTorch CNN for MNIST handwritten digit recognition.
Based on the original notebook implementation.
"""

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    Optimized for 28x28 grayscale digit images.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # input 1 channel, output 32 channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),  # 10 classes (digits 0-9)
        )

    def forward(self, x):
        return self.model(x)


# Legacy class name for compatibility
class DigitRecognitionCNN(ImageClassifier):
    """Alias for ImageClassifier to maintain compatibility."""

    pass


def get_device():
    """Get available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path="models/digit_recognizer.pth"):
    """Save model state dict."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path="models/digit_recognizer.pth", device="cpu"):
    """Load model from state dict."""
    model = ImageClassifier()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
