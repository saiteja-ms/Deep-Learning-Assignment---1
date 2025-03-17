import argparse
import wandb
import sys
import os
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.data import load_data
from src.model import NeuralNetwork
from src.visualization import plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments on MNIST dataset')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='mnist_experiments', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='teja_sai', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load MNIST data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data('mnist')
    
    # Define configurations based on Fashion-MNIST learnings
    # These are the top 3 configurations we recommend based on our experiments
    configs = [
        {
            "name": "config1_adam_relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_layers": 3,
            "hidden_size": 128,
            "activation": "ReLU",
            "weight_init": "Xavier",
            "weight_decay": 0.0005
        },
        {
            "name": "config2_nadam_relu",
            "optimizer": "nadam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_layers": 2,
            "hidden_size": 256,
            "activation": "ReLU",
            "weight_init": "Xavier",
            "weight_decay": 0
        },
        {
            "name": "config3_rmsprop_tanh",
            "optimizer": "rmsprop",
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_layers": 2,
            "hidden_size": 512,
            "activation": "tanh",
            "weight_init": "Xavier",
            "weight_decay": 0
        }
    ]
    
    results = {}
    
    for config in configs:
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=config["name"],
            config={
                "dataset": "mnist",
                "loss": "cross_entropy",
                "epochs": 10,
                **config
            },
            reinit=True
        )
        
        # Create model
        input_size = X_train.shape[1]
        hidden_sizes = [config["hidden_size"]] * config["num_layers"]
        output_size = 10
        
        model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=config["activation"],
            weight_init=config["weight_init"]
        )
        
        # Configure optimizer
        optimizer_config = {
            'learning_rate': config["learning_rate"],
            'momentum': 0.9,
            'beta': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': config["weight_decay"]
        }
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=10,
            batch_size=config["batch_size"],
            loss_name="cross_entropy",
            optimizer_name=config["optimizer"],
            optimizer_config=optimizer_config,
            wandb_log=True
        )
        
        # Evaluate on test set and plot confusion matrix
        test_accuracy = plot_confusion_matrix(model, X_test, y_test, 'mnist')
        
        # Store results
        results[config["name"]] = {
            'config': config,
            'history': history,
            'test_accuracy': test_accuracy
        }
        
        # Finish wandb run
        wandb.finish()
    
    # Print results summary
    print("\nMNIST Experiment Results:")
    print("-" * 50)
    for name, data in results.items():
        print(f"Configuration: {name}")
        print(f"Test Accuracy: {data['test_accuracy']:.4f}")
        print("-" * 50)
    
    # Create a summary report
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="mnist_summary",
        reinit=True
    )
    
    # Log a comparison table
    table_data = []
    for name, data in results.items():
        table_data.append([
            name, 
            data['config']['optimizer'],
            data['config']['activation'],
            data['config']['num_layers'],
            data['config']['hidden_size'],
            data['test_accuracy']
        ])
    
    comparison_table = wandb.Table(
        columns=["Configuration", "Optimizer", "Activation", "Num Layers", "Hidden Size", "Test Accuracy"],
        data=table_data
    )
    wandb.log({"mnist_comparison": comparison_table})
    
    # Create a bar chart comparing test accuracies
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    accuracies = [data['test_accuracy'] for data in results.values()]
    plt.bar(names, accuracies)
    plt.xlabel('Configuration')
    plt.ylabel('Test Accuracy')
    plt.title('MNIST Test Accuracy Comparison')
    plt.ylim(0.9, 1.0)  # MNIST accuracies are typically high
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('mnist_accuracy_comparison.png')
    
    # Log the comparison chart
    wandb.log({"mnist_accuracy_comparison": wandb.Image('mnist_accuracy_comparison.png')})
    
    wandb.finish()
    
    return results

if __name__ == "__main__":
    main()
