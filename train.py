import argparse
import numpy as np
import wandb
from neural_networks import NeuralNetwork
from keras.datasets import fashion_mnist, mnist
from data_loading import load_fashion_mnist
from wandb_ import train_with_wandb  # Import the train_with_wandb function
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Loss_function import cross_entropy_loss, mean_squared_error

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network with backpropagation')

    parser.add_argument('-wp', '--wandb_project', default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function for training')
    parser.add_argument('-o', '--optimizer', default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', default='random', choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4,
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', default='sigmoid',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function for hidden layers')

    return parser.parse_args()

def load_dataset(dataset):
    (X_train, y_train_one_hot, y_train), (X_val, y_val_one_hot, y_val), (X_test, y_test_one_hot, y_test) = load_fashion_mnist()
    X_train = (X_train, y_train_one_hot, y_train)
    X_val = (X_val, y_val_one_hot, y_val)
    X_test = (X_test, y_test_one_hot, y_test)  

    return  X_train, X_val, X_test

def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(filename)  # Save the figure
    plt.close()  # Close the figure

def main():
    args = parse_arguments()

    # Load dataset
    X_train, X_val, X_test = load_dataset(args.dataset)
    X_train, y_train_one_hot, y_train = X_train
    X_val, y_val_one_hot, y_val = X_val
    X_test, y_test_one_hot, y_test = X_test

    # Define network architecture
    input_size = X_train.shape[1]  # 784 for both MNIST and Fashion-MNIST
    hidden_sizes = [args.hidden_size] * args.num_layers
    output_size = 10  # 10 classes for both MNIST and Fashion-MNIST

    # Create model
    model = NeuralNetwork(input_size, hidden_sizes, output_size,
                          activation=args.activation, weight_init=args.weight_init)

    # Call train_with_wandb to handle training and wandb logging
    train_with_wandb(model, X_train, y_train_one_hot, X_val, y_val_one_hot, vars(args))

    # Evaluate on test set
    test_pred = model.forward(X_test)
    y_pred_test = model.predict(X_test)
    test_accuracy = np.mean(y_pred_test == np.argmax(y_test_one_hot, axis=1))
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    class_names = [str(i) for i in range(10)]  # Assuming 10 classes
    plot_confusion_matrix(np.argmax(y_test_one_hot, axis=1), y_pred_test, class_names, "confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    wandb.finish()

if __name__ == '__main__':
    main()