import numpy as np

class Loss:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        # Convert y_true to one-hot encoding
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
            y_true_one_hot[np.arange(y_true.size), y_true] = 1
            y_true = y_true_one_hot
        return 0.5 * np.mean(np.sum((y_pred - y_true)**2, axis=1))
    
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
            y_true_one_hot[np.arange(y_true.size), y_true] = 1
            y_true = y_true_one_hot
        return y_pred - y_true
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        # Convert y_true to one-hot encoding if needed
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
            y_true_one_hot[np.arange(y_true.size), y_true] = 1
            y_true = y_true_one_hot
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros((y_true.size, y_pred.shape[1]))
            y_true_one_hot[np.arange(y_true.size), y_true] = 1
            y_true = y_true_one_hot
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred
    
    @staticmethod
    def get_loss(name):
        if name == "mean_squared_error":
            return Loss.mean_squared_error, Loss.mean_squared_error_derivative
        elif name == "cross_entropy":
            return Loss.cross_entropy, Loss.cross_entropy_derivative
        else:
            raise ValueError(f"Loss function {name} not supported")
