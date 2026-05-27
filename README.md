# Foundations of Deep Learning: Neural Network From Scratch

A purely mathematical implementation of a neural network built to demonstrate the core calculus and linear algebra behind artificial intelligence. This project avoids all high-level frameworks (like TensorFlow or PyTorch) to focus strictly on building the "brain" of a neural network from the ground up using **Python** and **NumPy**.

## 🧠 Core Concepts Mastered
This repository serves as a functional sandbox for the mathematical building blocks of machine learning:
* **Vectorization:** Processing entire batches of data simultaneously using matrix multiplication rather than slow `for` loops.
* **Forward Propagation:** Passing input features through weight matrices and biases to generate predictions.
* **Activation Functions:** Implementing custom mathematical filters to introduce non-linearity.
* **Backward Propagation:** Using the chain rule of calculus to calculate error gradients across multiple layers.
* **Gradient Descent:** Iteratively updating parameters to minimize the loss function and "train" the network.

---

## 📐 The Mathematical Engine

This project programmatically executes the following deep learning formulas:

| Component | Mathematical Operation | Purpose |
| :--- | :--- | :--- |
| **Linear Hypothesis** | $Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$ | Computes the raw weighted sum of inputs. |
| **ReLU Activation** | $A^{[1]} = \max(0, Z^{[1]})$ | Activates hidden neurons, filtering out negative values. |
| **Error Calculation** | $dZ^{[2]} = A^{[2]} - Y$ | Measures the difference between predictions and true labels. |
| **Weight Gradients** | $dW = \frac{1}{m} dZ \cdot A^{T}$ | Calculates how much each weight contributed to the error. |
| **Parameter Update** | $W = W - \alpha \cdot dW$ | Applies learning rate ($\alpha$) to adjust weights and learn. |

---

## 🚀 How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/Herryn71x/MNIST_Scratch.git)
   cd MNIST_Scratch    
