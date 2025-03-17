import argparse
import wandb
import numpy as np
from src.data import load_data
from src.model import NeuralNetwork
from src.visualization import plot_fashion_mnist_samples, plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on Fashion MNIST')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='fashion_mnist_base', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='teja_sai', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    
    # Training arguments
    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='Number of epochs to train neural network')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='Batch size used to train neural network')
    
    # Loss function
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', 
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function to use for training')
    
    # Optimizer arguments
    parser.add_argument('-o', '--optimizer', type=str, default='adam', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, 
                        help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, 
                        help='Momentum used by momentum and nag optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, 
                        help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, 
                        help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, 
                        help='Beta2 used by adam and nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8, 
                        help='Epsilon used by optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, 
                        help='Weight decay used by optimizers')
    
    # Model architecture arguments
    parser.add_argument('-w_i', '--weight_init', type=str, default='Xavier', 
                        choices=['random', 'Xavier'],
                        help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, 
                        help='Number of hidden layers used in feedforward neural network')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, 
                        help='Number of hidden neurons in a feedforward layer')
    parser.add_argument('-a', '--activation', type=str, default='ReLU', 
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function to use in hidden layers')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "beta": args.beta,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "weight_decay": args.weight_decay,
            "weight_init": args.weight_init,
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "activation": args.activation
        }
    )
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)
    
    # Plot dataset samples
    if args.dataset == 'fashion_mnist':
        plot_fashion_mnist_samples(X_train, y_train)
    
    # Create model
    input_size = X_train.shape[1]  # 784 for Fashion MNIST
    hidden_sizes = [args.hidden_size] * args.num_layers
    output_size = 10  # 10 classes for Fashion MNIST
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=args.activation,
        weight_init=args.weight_init
    )
    
    # Configure optimizer
    optimizer_config = {
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'beta': args.beta,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'epsilon': args.epsilon,
        'weight_decay': args.weight_decay
    }
    
    # Train model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        optimizer_config=optimizer_config,
        wandb_log=True
    )
    
    # Evaluate on test set and plot confusion matrix
    test_accuracy = plot_confusion_matrix(model, X_test, y_test, args.dataset)
    
    # Finish wandb run
    wandb.finish()
    
    return model, history, test_accuracy

if __name__ == "__main__":
    main()
