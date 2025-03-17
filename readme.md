# Deep Learning Assignment 1: Feedforward Neural Network Implementation

## Links
- **GitHub Repository:** [Deep-Learning-Assignment-1](https://github.com/saiteja-ms/Deep-Learning-Assignment---1.git)
- **W&B Report:** [Fashion MNIST Sweep Backprop Report](https://wandb.ai/teja_sai-indian-institute-of-technology-madras/Fashion_mnist_sweep_backprop/reports/DA6401-Assignment-1-Report--VmlldzoxMTgyMDQ3NQ)


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

###Installation
Clone the repository:

```bash
git clone https://github.com/saiteja-ms/Deep-Learning-Assignment---1.git
```

Create a virtual environment and install dependencies:

```bash
conda create -n backprop python=3.8
conda activate backprop
pip install -r requirements.txt
```

### Set up Weights & Biases:

bash
wandb login
Scripts and Usage
1. Training a Model (train.py)
This script allows training a neural network with specified hyperparameters.

Key parameters:

--epochs: Number of training epochs

--batch_size: Batch size for training

--optimizer: Optimization algorithm (sgd, momentum, nag, rmsprop, adam, nadam)

--learning_rate: Learning rate for optimization

--num_layers: Number of hidden layers

--hidden_size: Number of neurons in each hidden layer

--activation: Activation function (identity, sigmoid, tanh, ReLU)

--weight_init: Weight initialization method (random, Xavier)

# Weights & Biases Integration and Model Training

## Setup
```bash
wandb login
```

### Explanation on the implementaion:
1. The backpropagation algorthm has been implemented that is flexible to accept both the number of hidden layers and number of units in those hidden layers as given by the user.
2. The optimizers(gradient descent, stochastic gradient descent,momentum, nag, rmsprop adam, nadam) have been implemented such that even their hyperparameters can be adjusted as wished.
3. Then, script for sweep has been done such that we have chosen bayes strategy and done 50 experiments.
## Scripts and Usage

### 1. Training a Model (`train.py`)
This script trains a neural network with specified hyperparameters.
for. eg. (you can give a similar set of hyperparameters to your train.py)
```bash
python train.py --epochs 10 --batch_size 16 --optimizer adam --learning_rate 0.001 --num_layers 5 --hidden_size 64 --activation ReLU --weight_init Xavier
```

**Key parameters:**
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--optimizer`: Optimization algorithm (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`)
- `--learning_rate`: Learning rate for optimization
- `--num_layers`: Number of hidden layers
- `--hidden_size`: Number of neurons in each hidden layer
- `--activation`: Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`)
- `--weight_init`: Weight initialization method (`random`, `Xavier`)

---

### 2. Hyperparameter Optimization (`run_sweep.py`)
This script runs a hyperparameter sweep using Weights & Biases to find the optimal configuration.

```bash
python experiments/run_sweep.py
```

**The sweep explores various combinations of:**
- **Hidden layers:** 3, 4, 5
- **Hidden sizes:** 32, 64, 128
- **Activation functions:** `sigmoid`, `tanh`, `ReLU`
- **Optimizers:** `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`
- **Learning rates:** `1e-4`, `1e-3`
- **Batch sizes:** 16, 32, 64
- **Weight initialization:** `random`, `Xavier`
- **Weight decay values:** 0, 0.0005, 0.5

---

### 3. Loss Function Comparison (`compare_loss.py`)
This script compares the performance of cross-entropy loss and mean squared error loss using the best hyperparameter configuration.

```bash
python experiments/compare_loss.py
```

---

### 5. MNIST Experiments (`mnist_experiments.py`)
This script applies the best configurations learned from Fashion-MNIST to the MNIST dataset.

```bash
python experiments/mnist_experiments.py
```

---

## Backpropagation Implementation
The backpropagation algorithm is implemented in the `backward` method of the `NeuralNetwork` class in `src/model.py`.

### Steps:
1. **Forward Pass**: Computes activations through each layer and stores them.
2. **Backward Pass**:
   - Computes output layer error using the derivative of the loss function.
   - Propagates the error backward, computing gradients for weights and biases:
     - `dW = (activation_prev.T @ delta) / batch_size`
     - `db = sum(delta, axis=0) / batch_size`
     - `delta_prev = (delta @ W.T) * activation_derivative(z_prev)`
3. **Parameter Update**: Uses optimizers (`SGD`, `Momentum`, `NAG`, `RMSprop`, `Adam`, `Nadam`) to update weights and biases.

This implementation supports **mini-batch gradient descent**.

---

## Key Findings

### Best Hyperparameter Configuration
- **Activation:** ReLU
- **Batch size:** 16
- **Number of hidden layers:** 5
- **Hidden size:** 64
- **Learning rate:** 0.001
- **Optimizer:** Adam
- **Weight initialization:** Xavier
- **Weight decay:** 0

**Performance:**
- **Validation accuracy:** 88.58%
- **Test accuracy:** 87.71%

### Hyperparameter Importance
- **Learning rate (0.001)** had the most significant impact on performance.
- **Adam optimizer** outperformed other optimizers consistently.
- **Deeper networks (5 layers)** achieved higher validation accuracies.
- **ReLU activation** performed better than `sigmoid` and `tanh`.
- **Batch size of 16** worked better than larger batch sizes.
- **Xavier initialization** was superior to random initialization.

### Loss Function Comparison
- **Cross-Entropy Loss**:
  - Validation Accuracy: **88.43%**
  - Test Accuracy: **87.53%**
- **Mean Squared Error Loss**:
  - Validation Accuracy: **88.05%**
  - Test Accuracy: **87.17%**

**Conclusion:** Cross-entropy performed slightly better, which is expected for classification tasks.

### MNIST Results
Applying the best configurations to MNIST:

| Optimizer | Activation | Layers | Validation Accuracy | Test Accuracy |
|-----------|-----------|--------|---------------------|--------------|
| Adam | ReLU | 5 | **97.85%** | **97.41%** |
| Nadam | ReLU | 4 | **97.97%** | **97.75%** |
| RMSprop | ReLU | 3 | **97.33%** | **97.47%** |

---

