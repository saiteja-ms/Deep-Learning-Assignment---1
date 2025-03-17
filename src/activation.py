import numpy as np

class Activation:
    """
    Class containing activation functions and their derivatives.
    """
    @staticmethod
    def identity(x):
        """
        Identity activation function: f(x) = x
        """
        return x
    
    @staticmethod
    def identity_derivative(x):
        """
        Derivative of identity activation function: f'(x) = 1
        """
        return np.ones_like(x)
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of sigmoid activation function: f'(x) = f(x) * (1 - f(x))
        """
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """
        Tanh activation function: f(x) = tanh(x)
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of tanh activation function: f'(x) = 1 - tanh^2(x)
        """
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def ReLU(x):
        """
        ReLU activation function: f(x) = max(0, x)
        """
        return np.maximum(0, x)
    
    @staticmethod
    def ReLU_derivative(x):
        """
        Derivative of ReLU activation function: f'(x) = 1 if x > 0 else 0
        """
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def softmax(x):
        """
        Softmax activation function: f(x_i) = exp(x_i) / sum(exp(x_j))
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def get_activation(activation_name):
        """
        Get activation function and its derivative by name.
        
        Args:
            activation_name (str): Name of activation function
            
        Returns:
            tuple: (activation_function, activation_derivative)
        """
        if activation_name == "identity":
            return Activation.identity, Activation.identity_derivative
        elif activation_name == "sigmoid":
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif activation_name == "tanh":
            return Activation.tanh, Activation.tanh_derivative
        elif activation_name == "ReLU":
            return Activation.ReLU, Activation.ReLU_derivative
        else:
            raise ValueError(f"Activation function {activation_name} not supported")
