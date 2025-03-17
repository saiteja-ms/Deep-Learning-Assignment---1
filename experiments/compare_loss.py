import argparse
import wandb
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_data
from src.model import NeuralNetwork
from src.visualization import plot_loss_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Compare loss functions for neural network')
    
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
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)
    
    # Define configurations
    loss_functions = ['cross_entropy', 'mean_squared_error']
    results = {}
    
    # Use the best model architecture from the hyperparameter sweep
    input_size = X_train.shape[1]  # 784 for Fashion MNIST
    hidden_sizes = [64] * 5  # 5 hidden layers with 64 neurons each
    output_size = 10  # 10 classes for Fashion MNIST
    
    for loss_fn in loss_functions:
        # Initialize wandb
        run_name = f"loss_comparison_{loss_fn}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "dataset": args.dataset,
                "loss": loss_fn,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 10,
                "batch_size": 16,
                "num_layers": 5,
                "hidden_size": 64,
                "activation": "ReLU",
                "weight_init": "Xavier",
                "weight_decay": 0
            },
            reinit=True
        )
        
        # Create model
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
            'weight_decay': 0
        }
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=10,
            batch_size=16,
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
            'loss': history['loss'],
            'accuracy': history['accuracy'],
            'val_loss': history['val_loss'],
            'val_accuracy': history['val_accuracy'],
            'test_accuracy': test_accuracy
        }
        
        # Finish wandb run
        wandb.finish()
    
    # Create comparison plots
    run_name = "loss_function_comparison"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        reinit=True
    )
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(results['cross_entropy']['loss'], label='Cross Entropy')
    plt.plot(results['mean_squared_error']['loss'], label='Mean Squared Error')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(results['cross_entropy']['val_loss'], label='Cross Entropy')
    plt.plot(results['mean_squared_error']['val_loss'], label='Mean Squared Error')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    wandb.log({"loss_comparison": wandb.Image('loss_comparison.png')})
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(results['cross_entropy']['accuracy'], label='Cross Entropy')
    plt.plot(results['mean_squared_error']['accuracy'], label='Mean Squared Error')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['cross_entropy']['val_accuracy'], label='Cross Entropy')
    plt.plot(results['mean_squared_error']['val_accuracy'], label='Mean Squared Error')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    wandb.log({"accuracy_comparison": wandb.Image('accuracy_comparison.png')})
    
    # Create a bar chart comparing test accuracies
    plt.figure(figsize=(8, 6))
    test_accuracies = [results['cross_entropy']['test_accuracy'], results['mean_squared_error']['test_accuracy']]
    plt.bar(['Cross Entropy', 'Mean Squared Error'], test_accuracies, color=['blue', 'orange'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, acc in enumerate(test_accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('test_accuracy_comparison.png')
    wandb.log({"test_accuracy_comparison": wandb.Image('test_accuracy_comparison.png')})
    
    # Log a summary table
    comparison_table = wandb.Table(
        columns=["Loss Function", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracy", "Test Accuracy"],
        data=[
            ["Cross Entropy", 
             results['cross_entropy']['loss'][-1], 
             results['cross_entropy']['val_loss'][-1], 
             results['cross_entropy']['accuracy'][-1], 
             results['cross_entropy']['val_accuracy'][-1], 
             results['cross_entropy']['test_accuracy']],
            ["Mean Squared Error", 
             results['mean_squared_error']['loss'][-1], 
             results['mean_squared_error']['val_loss'][-1], 
             results['mean_squared_error']['accuracy'][-1], 
             results['mean_squared_error']['val_accuracy'][-1], 
             results['mean_squared_error']['test_accuracy']]
        ]
    )
    wandb.log({"loss_function_comparison_table": comparison_table})
    
    # Print summary
    print("\nLoss Function Comparison Results:")
    print("-" * 50)
    print(f"Cross Entropy Test Accuracy: {results['cross_entropy']['test_accuracy']:.4f}")
    print(f"Mean Squared Error Test Accuracy: {results['mean_squared_error']['test_accuracy']:.4f}")
    print("-" * 50)
    
    wandb.finish()
    
    return results

if __name__ == "__main__":
    main()
