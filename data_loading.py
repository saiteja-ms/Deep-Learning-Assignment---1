from keras.datasets import fashion_mnist
import numpy as np

def load_fashion_mnist():
    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Reshape and normalize the dataset
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    # Create validation set (10% of training data)
    val_size = int(0.1 * X_train.shape[0])
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    # One-hot encode labels
    num_classes = 10
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1

    y_val_one_hot = np.zeros((y_val.shape[0], num_classes))
    y_val_one_hot[np.arange(y_val.shape[0]), y_val] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

    return (X_train, y_train_one_hot, y_train), (X_val, y_val_one_hot, y_val), (X_test, y_test_one_hot, y_test)
