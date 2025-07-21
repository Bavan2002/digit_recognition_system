# Handwritten Digit Recognition System

PyTorch-based CNN for recognizing handwritten digits using the MNIST dataset.

## Overview

This project implements a Convolutional Neural Network using PyTorch to classify handwritten digits (0-9). Achieves 99%+ accuracy on the MNIST benchmark.

## Features

- PyTorch CNN architecture with batch normalization
- Data augmentation (rotation, translation)
- Learning rate scheduling
- Interactive web interface
- Real-time digit recognition
- Confidence scores for all digits

## Installation

```bash
cd digit_recognition_system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train Model

```bash
python train.py
```

The MNIST dataset will be automatically downloaded.

### Launch Web App

```bash
streamlit run app.py
```

Upload images of handwritten digits for recognition.

## Model Architecture

- Input: 28x28 grayscale images
- 3 convolutional blocks (32→64→128 filters)
- Batch normalization after each conv layer
- MaxPooling and dropout for regularization
- 2 fully connected layers
- Output: 10 classes (digits 0-9)

## Performance

- Test Accuracy: 99.2%+
- Training time: ~10 minutes on GPU
- Inference: <10ms per image

## Dataset

MNIST (automatically downloaded):
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- Balanced classes (0-9)

## Technical Stack

- PyTorch
- torchvision
- Streamlit
- PIL, NumPy
- Python 3.8+

## Data Augmentation

- Random rotation (±10°)
- Random translation (10%)
- Normalization with MNIST statistics

## Training Features

- Adam optimizer
- ReduceLROnPlateau scheduler
- Best model checkpointing
- GPU acceleration support

## License

Educational and research purposes.
