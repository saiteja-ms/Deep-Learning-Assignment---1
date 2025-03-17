import numpy as np

class Loss:
    """
    Class containing loss functions and their derivatives.
    """
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Mean squared error loss function.
        
        Args:
            y_true (numpy.ndarray): Ground truth values
            y_pred (numpy.ndarray): Predicted values
            
        Returns:
            float: Mean squared error
        """
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    
    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        """
        Derivative of mean squared error loss function.
        
        Args:
            y_true (numpy.ndarray): Ground truth values
            y_pred (numpy.ndarray): Predicted values
            
        Returns:
            numpy.ndarray: Derivative of mean squared error
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def cross_entropy(y_true, y_pred, epsilon=1e-10):
        """
        Cross entropy loss function.
        
        Args:
            y_true (numpy.ndarray): Ground truth values
            y_pred (numpy.ndarray): Predicted values
            epsilon (float): Small value to avoid log(0)
            
        Returns:
            float: Cross entropy loss
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        """
        Derivative of cross entropy loss function with respect to softmax output.
        
        Args:
            y_true (numpy.ndarray): Ground truth values
            y_pred (numpy.ndarray): Predicted values
            
        Returns:
            numpy.ndarray: Derivative of cross entropy
        """
        return y_pred - y_true
    
    @staticmethod
    def get_loss(loss_name):
        """
        Get loss function and its derivative by name.
        
        Args:
            loss_name (str): Name of loss function
            
        Returns:
            tuple: (loss_function, loss_derivative)
        """
        if loss_name == "mean_squared_error":
            return Loss.mean_squared_error, Loss.mean_squared_error_derivative
        elif loss_name == "cross_entropy":
            return Loss.cross_entropy, Loss.cross_entropy_derivative
        else:
            raise ValueError(f"Loss function {loss_name} not supported")
