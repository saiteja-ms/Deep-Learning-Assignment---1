import argparse
import wandb
import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data
from src.model import NeuralNetwork
from src.visualization import plot_loss_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Compare loss functions for neural network')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='fashion_mnist_loss', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='teja_sai', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)
    
    # Define configurations
    loss_functions = ['cross_entropy', 'mean_squared_error']
    results = {}
    
    for loss_fn in loss_functions:
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"loss_{loss_fn}",
            config={
                "dataset": args.dataset,
                "loss": loss_fn,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 64,
                "num_layers": 3,
                "hidden_size": 128,
                "activation": "ReLU",
                "weight_init": "Xavier",
                "weight_decay": 0.0005
            },
            reinit=True
        )
        
        # Create model
        input_size = X_train.shape[1]
        hidden_sizes = [128] * 3
        output_size = 10
        
        model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation="ReLU",
            weight_init="Xavier"
        )
        
        # Configure optimizer
        optimizer_config = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0005
        }
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=10,
            batch_size=64,
            loss_name=loss_fn,
            optimizer_name="adam",
            optimizer_config=optimizer_config,
            wandb_log=True
        )
        
        # Evaluate on test set
        test_accuracy = model.evaluate(X_test, y_test)
        wandb.log({"test_accuracy": test_accuracy})
        
        # Store results
        results[loss_fn] = {
            'history': history,
            'test_accuracy': test_accuracy
        }
        
        # Finish wandb run
        wandb.finish()
    
    # Plot comparison
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="loss_comparison",
        reinit=True
    )
    plot_loss_comparison(results)
    wandb.finish()
    
    return results

if __name__ == "__main__":
    main()
