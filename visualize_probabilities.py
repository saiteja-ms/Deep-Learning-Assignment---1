import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import argparse
from src.data import load_data
from src.model import NeuralNetwork

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize class probabilities for Fashion-MNIST')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='Fashion_mnist_sweep_backprop', 
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='teja_sai-indian-institute-of-technology-madras', 
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
    
    # Dataset arguments
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for visualization')
    
    return parser.parse_args()

def visualize_class_probabilities(model, X_test, y_test, class_names):
    """
    Visualize the predicted probabilities for each class.
    
    Args:
        model: Trained neural network model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
        class_names: Names of the classes
    """
    # Get predictions
    y_pred_proba = model.forward(X_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Select a few correctly and incorrectly classified examples
    correct_indices = np.where(y_pred == y_true)[0]
    incorrect_indices = np.where(y_pred != y_true)[0]
    
    # Sample 5 correct and 5 incorrect predictions (if available)
    n_samples = 5
    correct_samples = np.random.choice(correct_indices, min(n_samples, len(correct_indices)), replace=False)
    incorrect_samples = np.random.choice(incorrect_indices, min(n_samples, len(incorrect_indices)), replace=False)
    
    # Combine samples
    sample_indices = np.concatenate([correct_samples, incorrect_samples])
    
    # Create figure for probability visualization
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(sample_indices):
        # Get the true class and predicted class
        true_class = y_true[idx]
        pred_class = y_pred[idx]
        
        # Get the probabilities for this example
        probs = y_pred_proba[idx]
        
        # Plot the probabilities
        plt.subplot(len(sample_indices), 1, i+1)
        bars = plt.bar(range(len(class_names)), probs, color='skyblue')
        
        # Highlight the true class and predicted class
        bars[true_class].set_color('green')
        if true_class != pred_class:
            bars[pred_class].set_color('red')
        
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.ylabel('Probability')
        
        # Add title indicating correct/incorrect prediction
        if true_class == pred_class:
            plt.title(f'Correct Prediction: True Class = {class_names[true_class]}', color='green')
        else:
            plt.title(f'Incorrect Prediction: True Class = {class_names[true_class]}, Predicted = {class_names[pred_class]}', color='red')
    
    plt.tight_layout()
    plt.savefig('class_probabilities.png')
    
    # Log to wandb
    wandb.log({"class_probabilities": wandb.Image('class_probabilities.png')})
    
    # Create confusion probability matrix (average probability per true class)
    plt.figure(figsize=(12, 10))
    
    # Initialize confusion probability matrix
    conf_proba = np.zeros((len(class_names), len(class_names)))
    
    # For each true class, compute average predicted probability for each class
    for i in range(len(class_names)):
        # Get indices of samples with true class i
        indices = np.where(y_true == i)[0]
        if len(indices) > 0:
            # Get average probabilities for these samples
            conf_proba[i] = np.mean(y_pred_proba[indices], axis=0)
    
    # Plot as a heatmap
    ax = sns.heatmap(conf_proba, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Average Predicted Probability per True Class')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_probability_matrix.png')
    
    # Log to wandb
    wandb.log({"confusion_probability_matrix": wandb.Image('confusion_probability_matrix.png')})

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize wandb
    run_name = f"visualize_probabilities_{args.dataset}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name
    )
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)
    
    # Define class names
    if args.dataset == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:  # mnist
        class_names = [str(i) for i in range(10)]
    
    # Create model with best parameters from hyperparameter sweep
    input_size = X_train.shape[1]  # 784 for Fashion MNIST
    hidden_sizes = [64] * 5  # 5 hidden layers with 64 neurons each (best config)
    output_size = 10  # 10 classes
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation="ReLU",  # Best activation
        weight_init="Xavier"  # Best weight initialization
    )
    
    # Configure optimizer (using best hyperparameters)
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
        batch_size=16,  # Best batch size
        loss_name="cross_entropy",  # Best loss function
        optimizer_name="adam",  # Best optimizer
        optimizer_config=optimizer_config,
        wandb_log=True
    )
    
    # Visualize probabilities
    visualize_class_probabilities(model, X_test, y_test, class_names)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
