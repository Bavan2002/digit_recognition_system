# Handwritten Digit Recognition System

PyTorch-based CNN for recognizing handwritten digits using the MNIST dataset.

## Overview

This project implements a Convolutional Neural Network using PyTorch to classify handwritten digits (0-9). Achieves 98%+ accuracy on the MNIST benchmark.

## Features

- PyTorch CNN architecture (3 convolutional layers)
- Interactive Streamlit web interface
- Real-time digit recognition
- Confidence scores for all digits
- Pre-trained model included
- Fast inference (<10ms per image)

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
cd digit_recognition_system
uv sync
```

### Using pip (Traditional)

```bash
cd digit_recognition_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Launch Web App

#### With uv:
```bash
uv run streamlit run app.py
```

#### With pip:
```bash
streamlit run app.py
```

The app will open in your browser. Upload images of handwritten digits for recognition.

### Train Model (Optional)

A pre-trained model is included. To train a new model:

#### With uv:
```bash
uv run python train.py
```

#### With pip:
```bash
python train.py
```

The MNIST dataset will be automatically downloaded.

## Model Architecture

- Input: 28x28 grayscale images
- 3 convolutional layers (32→64→64 filters)
- ReLU activation functions
- Flattening layer
- Fully connected output layer
- Output: 10 classes (digits 0-9)

## Performance

- Test Accuracy: 98.83%
- Training time: ~2-3 minutes on GPU, ~10 minutes on CPU
- Inference: <10ms per image

## Dataset

MNIST (automatically downloaded):
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- Balanced classes (0-9)

## Technical Stack

- Python 3.13+
- PyTorch 2.0+
- torchvision
- Streamlit
- PIL, NumPy
- uv package manager

## Project Structure

```
digit_recognition_system/
├── app.py              # Streamlit web interface
├── model.py            # CNN model definition
├── train.py            # Training script
├── models/             # Saved model weights
│   └── mnist_cnn_model.pt
├── data/               # MNIST dataset (auto-downloaded)
├── old/                # Original notebook and model
├── pyproject.toml      # uv/pip dependencies
└── README.md
```

## Tips for Best Results

- Use clear handwritten digits
- Single digit per image
- Black digit on white background (or vice versa)
- Centered digit
- Image will be automatically resized to 28x28

## License

Educational and research purposes.
