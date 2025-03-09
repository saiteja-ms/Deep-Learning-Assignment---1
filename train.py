import argparse
import numpy as np
import wandb
from neural_networks import NeuralNetwork
from keras.datasets import fashion_mnist
from data_loading import load_fashion_mnist
from wandb_ import train_with_wandb  # Import the train_with_wandb function
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Loss_function import cross_entropy_loss, mean_squared_error

def plot_sample_images(X_train, y_train, class_names):
    """Plots 1 sample image for each class in a grid."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # Create 2x5 grid of subplots
    axes = axes.ravel()  # Flatten the 2D array of axes into a 1D array

    for i in range(num_classes):
        # Find the first index of the class
        idx = np.where(y_train == i)[0][0]
        image = X_train[idx].reshape(28, 28)  # Reshape back to 28x28
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(class_names[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("fashion_mnist_samples.png") # Save the figure
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network with backpropagation')

    parser.add_argument('-wp', '--wandb_project', default='BackPropagation_implementation',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=5,
                        help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function for training')
    parser.add_argument('-o', '--optimizer', default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
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

    # Initialize wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    config = wandb.config

    # Load dataset
    X_train, X_val, X_test = load_dataset(config.dataset)
    X_train, y_train_one_hot, y_train = X_train
    X_val, y_val_one_hot, y_val = X_val
    X_test, y_test_one_hot, y_test = X_test

    # Plot sample images for Fashion-MNIST (Question 1)
    if config.dataset == 'fashion_mnist':

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        plot_sample_images(X_train, y_train, class_names)
        wandb.log({"fashion_mnist_samples": wandb.Image("fashion_mnist_samples.png")}) # Log to wandb

    # Define network architecture
    input_size = X_train.shape[1]  # 784 for both MNIST and Fashion-MNIST
    hidden_sizes = [config.hidden_size] * config.num_layers
    output_size = 10  # 10 classes for both MNIST and Fashion-MNIST

    # Create model
    model = NeuralNetwork(input_size, hidden_sizes, output_size,
                          activation=config.activation, weight_init=config.weight_init)

    # Define optimizer parameters
    optimizer_params = {
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'momentum': config.momentum,
        'beta': config.beta,
        'beta1': config.beta1,
        'beta2': config.beta2,
        'epsilon': config.epsilon
    }
    
    # Call train_with_wandb to handle training and wandb logging
    train_with_wandb(model, X_train, y_train_one_hot, X_val, y_val_one_hot, config)

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