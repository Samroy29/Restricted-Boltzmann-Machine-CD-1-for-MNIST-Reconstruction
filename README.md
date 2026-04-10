
# Restricted Boltzmann Machine (RBM) for MNIST Reconstruction

This project implements a **Restricted Boltzmann Machine (RBM)** from scratch using NumPy.  
The RBM is trained on the MNIST dataset and used as a simple **unsupervised autoencoder** to reconstruct handwritten digits.

---

## 🧠 What is a Restricted Boltzmann Machine?

A Restricted Boltzmann Machine is a probabilistic neural network consisting of:

- A **visible layer** (input data)
- A **hidden layer** (latent feature representation)

There are no connections within a layer, only between visible and hidden units.

The model learns a joint probability distribution:

\[
P(v, h)
\]

where:
- \(v\) = visible units (input images)
- \(h\) = hidden units (latent features)

---

## ⚙️ How the Model Works

The RBM is trained using **Contrastive Divergence (CD-1)**:

### 1. Positive Phase
The hidden layer is activated given the input:

\[
P(h|v) = \sigma(Wv + c)
\]

### 2. Negative Phase (Reconstruction)
The model reconstructs the input:

\[
P(v|h) = \sigma(W^T h + b)
\]

### 3. Parameter Update
Weights are updated using:

\[
\Delta W = v_0 h_0^T - v_1 h_1^T
\]

---

## 📊 Dataset

- MNIST handwritten digits
- Images are:
  - Normalized to [0,1]
  - Flattened to 784-dimensional vectors
  - Binarized for RBM training

---

## 🏗 Model Architecture

- Visible units: 784 (28×28 pixels)
- Hidden units: 100
- Activation function: Sigmoid
- Sampling: Bernoulli sampling
- Training method: CD-1

---

## 📈 Results

The model is able to reconstruct MNIST digits after training.

Example:

- Original digits → reconstructed digits (visible similarity after training)
- Loss decreases over epochs

---

## 🖼 Example Output

(Add your generated plots here)

- Training vs Validation Loss
- Original vs Reconstructed digits

---

## 🚀 How to Run

```bash
pip install numpy matplotlib keras
python main.py
