import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def mean_squared_error(y_true, y_pred):
    return np.mean(np.sum((y_true - y_pred)**2, axis=1))

def cross_entropy_gradient(y_true, y_pred):
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / y_true.shape[0]

def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
