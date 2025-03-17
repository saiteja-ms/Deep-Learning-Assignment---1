# Deep Learning Assignment 1: Feedforward Neural Network Implementation

This repository contains the implementation of a feedforward neural network from scratch using NumPy, along with backpropagation for training the network on the Fashion-MNIST dataset.

## Repository Structure

```
├── src/
│   ├── activation.py       # Activation functions and their derivatives
│   ├── data.py             # Data loading and preprocessing functions
│   ├── loss.py             # Loss functions and their derivatives
│   ├── model.py            # Neural network implementation
│   ├── optimizers.py       # Optimization algorithms
│   └── visualization.py    # Visualization utilities
├── experiments/
│   ├── compare_loss.py     # Compare cross-entropy and MSE loss
│   ├── mnist_experiments.py # Apply learnings to MNIST dataset
│   └── run_sweep.py        # Hyperparameter optimization with wandb
├── train.py                # Main training script
├── requirements.txt        # Required packages
└── README.md               # Project documentation
```

## Installation

Clone the repository:

```bash
git clone https://github.com/saiteja-ms/Deep-Learning-Assignment---1.git
cd Deep-Learning-Assignment---1
```

Create a virtual environment and install dependencies:

```bash
conda create -n backprop python=3.8
conda activate backprop
pip install -r requirements.txt
```

## Set up Weights & Biases

```bash
wandb login
```

## Implementation Details

### Data Loading and Preprocessing (`src/data.py`)

The `data.py` module handles loading and preprocessing the Fashion-MNIST and MNIST datasets:

- **Loading Data**: Uses Keras datasets to load Fashion-MNIST or MNIST.
- **Preprocessing**:
  - Normalizes pixel values by dividing by 255.0.
  - Reshapes 28x28 images to 784-dimensional vectors.
  - Converts labels to one-hot encoding (10 classes).
  - Splits training data into training and validation sets (90% train, 10% validation).

```python
def load_data(dataset='fashion_mnist'):
    # Load dataset (fashion_mnist or mnist)
    # Normalize pixel values to [0,1]
    # Reshape images to vectors
    # Convert labels to one-hot encoding
    # Split into train, validation, and test sets
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

### Activation Functions (`src/activation.py`)

The `activation.py` module implements various activation functions and their derivatives:

- **Identity**: `f(x) = x`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Tanh**: `f(x) = tanh(x)`
- **ReLU**: `f(x) = max(0, x)`
- **Softmax**: For output layer, with numerical stability improvements.

Each activation function has a corresponding derivative function used during backpropagation.

### Loss Functions (`src/loss.py`)

The `loss.py` module implements loss functions and their derivatives:

- **Mean Squared Error**: For regression problems.
- **Cross-Entropy**: For classification problems, with numerical stability improvements.

```python
def cross_entropy(y_true, y_pred, epsilon=1e-10):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

### Neural Network Model (`src/model.py`)

The `model.py` module contains the core `NeuralNetwork` class with the following components:

#### Initialization:
- Sets up network architecture with configurable hidden layers.
- Supports different weight initialization methods (random, Xavier).
- Initializes weights and biases for all layers.

#### Forward Propagation:
- Computes activations through the network.
- Stores intermediate activations and pre-activations for backpropagation.
- Applies appropriate activation function at each layer.
- Uses softmax for the output layer.

```python
def forward(self, X):
    # Store activations and pre-activations for backpropagation
    self.a_values = [X]  # Input layer activation
    
    # Forward pass through hidden layers
    for i in range(len(self.weights) - 1):
        z = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
        self.z_values.append(z)
        a = self.activation(z)
        self.a_values.append(a)
    
    # Output layer (softmax activation for classification)
    z_out = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
    self.z_values.append(z_out)
    a_out = self.softmax(z_out)
    self.a_values.append(a_out)
    
    return a_out
```

#### Backpropagation:
- Computes gradients for all weights and biases.
- Starts with output layer error based on loss derivative.
- Propagates error backward through the network.
- Computes weight and bias gradients for each layer.

```python
def backward(self, X, y, y_pred, loss_derivative):
    batch_size = X.shape[0]
    
    # Initialize gradients
    dw = [np.zeros_like(w) for w in self.weights]
    db = [np.zeros_like(b) for b in self.biases]
    
    # Output layer error
    delta = loss_derivative(y, y_pred)
    
    # Backpropagate through layers
    for l in range(len(self.weights) - 1, -1, -1):
        # Compute gradients for weights and biases
        dw[l] = np.dot(self.a_values[l].T, delta) / batch_size
        db[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # Backpropagate error to previous layer (if not input layer)
        if l > 0:
            delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(self.z_values[l-1])
    
    return dw, db
```
