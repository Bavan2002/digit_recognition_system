# MNIST Handwritten Digits Classification using CNN (PyTorch)

**Jupyter Notebook:** [Open Notebook](https://colab.research.google.com/drive/1K0fwgnNoSipwTG1xtwSJOIaUmilcT5UB?usp=sharing) 

## Overview
This project demonstrates building, training, and evaluating a **Convolutional Neural Network (CNN)** on the MNIST dataset using **PyTorch**. The model can classify handwritten digits (0-9) and supports predictions on custom images.

## Tech Stack
- Python 3.x  
- PyTorch  
- Torchvision  
- Matplotlib  
- Pillow  
- Jupyter Notebook  

## Features
- Train a CNN for handwritten digit classification  
- Save and load trained model weights  
- Predict custom images via interactive upload in the notebook  
- Step-by-step notebook for learning and experimentation  

## How to Run

### Jupyter Notebook
```bash
jupyter notebook mnist_cnn_notebook.ipynb
````

### Python Script

```bash
python torchnn.py
```

Upload a 28x28 grayscale image when prompted to get predictions.

## Model

The trained PyTorch model weights are saved in `model_state.pt` for easy reuse and inference.

```
