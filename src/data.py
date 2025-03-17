import numpy as np
from keras.datasets import fashion_mnist, mnist

def load_data(dataset='fashion_mnist'):
    """
    Load and preprocess the dataset.
    
    Args:
        dataset (str): Dataset to load ('fashion_mnist' or 'mnist')
        
    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    # Load dataset
    if dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Split training data to create validation set
    val_size = int(0.1 * X_train.shape[0])
    indices = np.random.permutation(X_train.shape[0])
    X_val = X_train[indices[:val_size]]
    y_val = y_train[indices[:val_size]]
    X_train = X_train[indices[val_size:]]
    y_train = y_train[indices[val_size:]]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
