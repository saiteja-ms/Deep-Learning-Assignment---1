import numpy as np
from optimizers import SGD, Momentum, NAG, RMSprop, Adam

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init

        # Initialize weights
        if weight_init == 'random':
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == 'Xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        else:
            raise ValueError(f"Unsupported weight initialization method: {weight_init}")

        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
        self.z = None
        self.dweights = None
        self.dbiases = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self._apply_activation(self.z)
        return self.output

    def backward(self, delta, learning_rate):
        batch_size = delta.shape[0]
        self.dweights = np.dot(self.input.T, delta) / batch_size
        self.dbiases = np.sum(delta, axis=0, keepdims=True) / batch_size

        # Compute gradient for previous layer
        if self.activation != 'softmax':
            delta_prev = np.dot(delta, self.weights.T) * self._activation_derivative(self.z)
        else:
            delta_prev = np.dot(delta, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

        return delta_prev

    def _apply_activation(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'ReLU':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            sig = self._apply_activation(z)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        elif self.activation == 'ReLU':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'identity':
            return np.ones_like(z)
        else:
            raise ValueError(f"Derivative not implemented for activation: {self.activation}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        self.layers = []

        # Input to first hidden layer
        self.layers.append(Layer(input_size, hidden_sizes[0], activation, weight_init))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], activation, weight_init))

        # Output layer
        self.layers.append(Layer(hidden_sizes[-1], output_size, activation='softmax', weight_init=weight_init))

    def forward(self, X):
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, y, learning_rate, optimizer):
        delta = y
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
            optimizer.update(layer)

    def train(self, X, y, epochs, batch_size, learning_rate, optimizer_name='sgd', **optimizer_params):
        num_samples = X.shape[0]

        optimizer_classes = {
            'sgd': SGD,
            'momentum': Momentum,
            'nag': NAG,
            'rmsprop': RMSprop,
            'adam': Adam
        }

        if optimizer_name not in optimizer_classes:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer = optimizer_classes[optimizer_name](learning_rate=learning_rate, **optimizer_params)

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)
                self.backward(y_batch, learning_rate, optimizer)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
