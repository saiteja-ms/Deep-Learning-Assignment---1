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
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'mnist':
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    # Split training data to create validation set
    val_size = int(0.1 * X_train_full.shape[0])
    X_val = X_train_full[:val_size]
    y_val = y_train_full[:val_size]
    X_train = X_train_full[val_size:]
    y_train = y_train_full[val_size:]
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.zeros((y_train.size, num_classes))
    y_train_onehot[np.arange(y_train.size), y_train] = 1
    
    y_val_onehot = np.zeros((y_val.size, num_classes))
    y_val_onehot[np.arange(y_val.size), y_val] = 1
    
    y_test_onehot = np.zeros((y_test.size, num_classes))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    return (X_train, y_train_onehot), (X_val, y_val_onehot), (X_test, y_test_onehot)
