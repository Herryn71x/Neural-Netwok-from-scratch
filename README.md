# Foundations of Deep Learning: Neural Network From Scratch

A purely mathematical implementation of a neural network built to demonstrate the core calculus and linear algebra behind artificial intelligence. This project avoids all high-level frameworks (like TensorFlow, PyTorch, or Keras) and implements everything from first principles using NumPy.

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

## 📦 Installation & Dependencies

### Requirements
* Python 3.8+
* NumPy
* Pandas (for data handling)
* Matplotlib (for visualization)

### Setup
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib
```

---

## 🚀 How to Run

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Herryn71x/Neural-Netwok-from-scratch.git
   cd Neural-Netwok-from-scratch
   ```

2. Install dependencies (see Installation section above)

3. Run the main neural network training script:
   ```bash
   python train.py
   ```

4. View results and visualizations in the output directory

---

## 📁 Project Structure

```
Neural-Netwok-from-scratch/
├── README.md
├── train.py                 # Main training script
├── neural_network.py        # Core neural network implementation
├── activation.py            # Activation functions (ReLU, Sigmoid, etc.)
├── loss.py                  # Loss functions
├── utils.py                 # Utility functions (data preprocessing, etc.)
├── data/                    # Dataset directory
│   └── data.csv
└── results/                 # Output directory for trained models
    ├── model_weights.pkl
    └── training_plots.png
```

---

## 💡 Usage Example

```python
from neural_network import NeuralNetwork
import numpy as np

# Initialize network: input layer (784) → hidden layer (128) → output layer (10)
nn = NeuralNetwork(layers=[784, 128, 10], activation='relu')

# Load your data
X_train, y_train = load_data('data/data.csv')

# Train the network
nn.train(X_train, y_train, epochs=50, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X_test)
```

---

## 📊 Results

After training on a standard dataset:
* **Accuracy:** [Add your results here]
* **Training Time:** [Add your results here]
* **Final Loss:** [Add your results here]

See `results/` directory for detailed plots and visualizations.

---

## 🔧 Key Components

### Neural Network Class
Implements the full architecture with configurable layers and activation functions.

### Backpropagation Algorithm
Uses automatic differentiation via the chain rule to compute gradients for all parameters.

### Gradient Descent Optimizer
Updates weights and biases iteratively to minimize loss function.

### Activation Functions
- **ReLU:** Rectified Linear Unit for hidden layers
- **Sigmoid:** For binary classification output
- **Softmax:** For multi-class classification output

---

## 🎓 Learning Resources

This project is ideal for understanding:
* How neural networks learn through gradient descent
* The mathematical foundation of deep learning
* Manual implementation of forward and backward passes
* Matrix operations in machine learning

---

## 📝 License

This project is open source and available under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
* Submit issues for bugs or suggestions
* Create pull requests with improvements
* Suggest new features or optimizations

---

## 📧 Contact

For questions or discussions about this project, please open an issue on GitHub.

---

**Built with ❤️ to understand the math behind AI**
