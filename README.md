# Restricted Boltzmann Machine (RBM) – MNIST Reconstruction

This project implements a **Restricted Boltzmann Machine (RBM)** from scratch using NumPy.  
The model is trained on the MNIST dataset to reconstruct handwritten digits.

---

## 🧠 Project Goal

The goal of this project is to learn how a simple **unsupervised neural network** can:
- learn patterns in handwritten digits
- compress input data into a hidden representation
- reconstruct the original input

---

## 📊 Dataset

- MNIST handwritten digits
- Images are:
  - normalized to [0,1]
  - flattened (28×28 → 784 features)
  - binarized for training

---

## 🏗 Model Overview

- Visible layer: 784 neurons
- Hidden layer: 100 neurons
- Activation: Sigmoid
- Sampling: Bernoulli sampling

---

## 🔁 Training

The model is trained using an iterative process where it:
- encodes input into hidden features
- reconstructs the input
- updates weights based on reconstruction error

Training uses a simple contrastive learning approach.

---

## 📈 Results

After training, the model is able to:
- reconstruct MNIST digits
- preserve the general shape of digits
- reduce reconstruction error over time

---

## 🖼 Example Output

Include here:
- training vs validation loss plot
- original vs reconstructed digits

---

## 🚀 How to Run

```bash
pip install numpy matplotlib keras
python main.py
