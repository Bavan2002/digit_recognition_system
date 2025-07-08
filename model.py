"""
PyTorch CNN for MNIST handwritten digit recognition.
Implements efficient architecture for digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecognitionCNN(nn.Module):
 """
 Convolutional Neural Network for MNIST digit classification.
 Optimized for 28x28 grayscale digit images.
 """

 def __init__(self):
 super(DigitRecognitionCNN, self).__init__()

 # Convolutional layers
 self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
 self.bn1 = nn.BatchNorm2d(32)

 self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
 self.bn2 = nn.BatchNorm2d(64)

 self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
 self.bn3 = nn.BatchNorm2d(128)

 # Pooling
 self.pool = nn.MaxPool2d(2, 2)

 # Dropout
 self.dropout1 = nn.Dropout(0.25)
 self.dropout2 = nn.Dropout(0.5)

 # Fully connected layers
 self.fc1 = nn.Linear(128 * 3 * 3, 256)
 self.fc2 = nn.Linear(256, 10)

 def forward(self, x):
 # Block 1
 x = self.conv1(x)
 x = self.bn1(x)
 x = F.relu(x)
 x = self.pool(x)
 x = self.dropout1(x)

 # Block 2
 x = self.conv2(x)
 x = self.bn2(x)
 x = F.relu(x)
 x = self.pool(x)
 x = self.dropout1(x)

 # Block 3
 x = self.conv3(x)
 x = self.bn3(x)
 x = F.relu(x)
 x = self.pool(x)
 x = self.dropout1(x)

 # Flatten
 x = x.view(-1, 128 * 3 * 3)

 # Dense layers
 x = self.fc1(x)
 x = F.relu(x)
 x = self.dropout2(x)
 x = self.fc2(x)

 return x


def get_device():
 """Get available device (CUDA or CPU)."""
 return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path="models/digit_recognizer.pth"):
 """Save model state dict."""
 torch.save(model.state_dict(), path)
 print(f"Model saved to {path}")


def load_model(path="models/digit_recognizer.pth", device="cpu"):
 """Load model from state dict."""
 model = DigitRecognitionCNN()
 model.load_state_dict(torch.load(path, map_location=device))
 model.eval()
 return model
