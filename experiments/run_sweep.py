import argparse
import wandb
import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data
from src.model import NeuralNetwork

def sweep_train():
    """
    Training function for wandb sweep.
    """
    # Initialize wandb first
    wandb.init()
    
    # Access hyperparameters after initialization
    config = wandb.config
    
    # Create a meaningful name based on hyperparameters and update the run name
    run_name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}"
    wandb.run.name = run_name
    wandb.run.save()
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(config.dataset)
    # Create model
    input_size = X_train.shape[1]  # 784 for Fashion MNIST
    hidden_sizes = [config.hidden_size] * config.num_layers
    output_size = 10  # 10 classes for Fashion MNIST
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=config.activation,
        weight_init=config.weight_init
    )
    
    # Configure optimizer
    optimizer_config = {
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    
    # Train model
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.epochs,
        batch_size=config.batch_size,
        loss_name=config.loss,
        optimizer_name=config.optimizer,
        optimizer_config=optimizer_config,
        wandb_log=True
    )
    
    # Evaluate on test set
    test_accuracy = model.evaluate(X_test, y_test)
    wandb.log({"test_accuracy": test_accuracy})

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep for neural network')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='Fashion_mnist_sweep_backprop', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='teja_sai-indian-institute-of-technology-madras', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'dataset': {'value': args.dataset},
            'epochs': {'values': [5, 10]},
            'batch_size': {'values': [16, 32, 64]},
            'num_layers': {'values': [3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'activation': {'values': ['sigmoid', 'tanh', 'ReLU']},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'learning_rate': {'values': [1e-4, 1e-3]},
            'weight_decay': {'values': [0, 0.0005, 0.5]},
            'weight_init': {'values': ['random', 'Xavier']},
            'loss': {'value': 'cross_entropy'}
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_train, count=50)  # Run 50 experiments

if __name__ == "__main__":
    main()
    
