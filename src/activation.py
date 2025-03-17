import numpy as np

class Activation:
    @staticmethod
    def identity(x):
        return x
    
    @staticmethod
    def identity_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_x = Activation.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def get_activation(name):
        if name == "identity":
            return Activation.identity, Activation.identity_derivative
        elif name == "sigmoid":
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif name == "tanh":
            return Activation.tanh, Activation.tanh_derivative
        elif name == "ReLU":
            return Activation.relu, Activation.relu_derivative
        else:
            raise ValueError(f"Activation function {name} not supported")
