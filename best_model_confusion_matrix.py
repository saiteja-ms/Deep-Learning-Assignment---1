import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
from src.data import load_data
from src.model import NeuralNetwork

def plot_creative_confusion_matrix(model, X_test, y_test):
    """
    Create a creative confusion matrix visualization for the best model.
    
    Args:
        model: Trained neural network model
        X_test: Test data
        y_test: Test labels (one-hot encoded)
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create a more creative visualization
    plt.figure(figsize=(14, 12))
    
    # Use a custom colormap with gradient
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Plot the normalized confusion matrix with correct format for floats
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names)
    
    # Add title with model details
    plt.title('Fashion-MNIST Classification Results\n5-layer ReLU Network with Adam Optimizer', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Add axis labels with better styling
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Calculate and display accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Highlight the diagonal (correct predictions)
    for i in range(len(class_names)):
        plt.text(i+0.5, i+0.5, f'{cm_normalized[i,i]:.2f}', 
                ha='center', va='center', fontsize=15, color='white', 
                bbox={'facecolor':'green', 'alpha':0.6, 'pad':10})
    
    # Find and highlight the most confused classes
    np.fill_diagonal(cm_normalized, 0)  # Remove diagonal for finding max confusion
    max_confusion = np.unravel_index(np.argmax(cm_normalized), cm_normalized.shape)
    plt.text(max_confusion[1]+0.5, max_confusion[0]+0.5, f'{cm_normalized[max_confusion]:.2f}', 
            ha='center', va='center', fontsize=15, color='white', 
            bbox={'facecolor':'red', 'alpha':0.6, 'pad':10})
    
    # Add model architecture details
    arch_details = "Model Architecture:\n" + \
                   "- 5 Hidden Layers (64 neurons each)\n" + \
                   "- ReLU Activation\n" + \
                   "- Adam Optimizer (lr=0.001)\n" + \
                   "- Batch Size: 16\n" + \
                   "- Xavier Initialization\n" + \
                   "- No Weight Decay"
    
    plt.figtext(0.15, 0.02, arch_details, fontsize=10,
                bbox={'facecolor':'lightblue', 'alpha':0.5, 'pad':5, 'boxstyle':'round'})
    
    plt.tight_layout(rect=[0, 0.05, 0.8, 0.95])
    plt.savefig('best_model_confusion_matrix_normalized.png')
    
    # Log to wandb with a meaningful name
    run_name = f"cm_normalized_hl_5_bs_16_ac_ReLU"
    wandb.init(name=run_name, project="Fashion_mnist_sweep_backprop", entity="teja_sai-indian-institute-of-technology-madras")
    wandb.log({"best_model_confusion_matrix_normalized": wandb.Image('best_model_confusion_matrix_normalized.png')})
    wandb.finish()
    
    return accuracy

def main():
    # Initialize wandb first
    run_name = f"train_hl_5_bs_16_ac_ReLU"
    wandb.init(name=run_name, project="Fashion_mnist_sweep_backprop", entity="teja_sai-indian-institute-of-technology-madras")
    
    # Load and preprocess data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data('fashion_mnist')
    
    # Create model with best parameters from the image
    input_size = X_train.shape[1]  # 784 for Fashion MNIST
    hidden_sizes = [64] * 5  # 5 hidden layers with 64 neurons each
    output_size = 10  # 10 classes for Fashion MNIST
    
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
        loss_name="cross_entropy",
        optimizer_name="adam",
        optimizer_config=optimizer_config,
        wandb_log=True
    )
    
    # Finish the training wandb run
    wandb.finish()
    
    # Evaluate on test set and create confusion matrix (this will create a new wandb run)
    test_accuracy = plot_creative_confusion_matrix(model, X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
