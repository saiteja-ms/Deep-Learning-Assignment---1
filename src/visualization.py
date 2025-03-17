import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

def plot_fashion_mnist_samples(X_train, y_train):
    """
    Plot one sample from each class in Fashion-MNIST dataset.
    
    Args:
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
    """
    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Plot one sample from each class
    plt.figure(figsize=(10, 10))
    for i in range(10):
        # Find first instance of class i
        idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]
        
        # Plot
        plt.subplot(3, 4, i+1)
        plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png')
    
    # Log to wandb
    wandb.log({"fashion_mnist_samples": wandb.Image('fashion_mnist_samples.png')})

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model (NeuralNetwork): Trained neural network model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
        
    Returns:
        float: Test accuracy
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create a more creative visualization
    plt.figure(figsize=(12, 10))
    
    # Plot the confusion matrix with a custom colormap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                    xticklabels=class_names, yticklabels=class_names)
    
    # Add title and labels with custom styling
    plt.title('Confusion Matrix', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add accuracy information
    accuracy = np.mean(y_pred == y_true)
    plt.figtext(0.5, 0.01, f'Test Accuracy: {accuracy:.4f}', 
                ha='center', fontsize=16, bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':5})
    
    # Highlight the diagonal (correct predictions)
    for i in range(len(class_names)):
        plt.text(i+0.5, i+0.5, f'{cm[i,i]}', 
                ha='center', va='center', fontsize=15, color='white', 
                bbox={'facecolor':'green', 'alpha':0.6, 'pad':10})
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Log to wandb
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    
    return accuracy

def plot_loss_comparison(cross_entropy_results, mse_results):
    """
    Plot comparison of loss functions.
    
    Args:
        cross_entropy_results (dict): Results from cross-entropy loss
        mse_results (dict): Results from mean squared error loss
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(cross_entropy_results['loss'], label='Cross Entropy')
    plt.plot(mse_results['loss'], label='Mean Squared Error')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(cross_entropy_results['val_loss'], label='Cross Entropy')
    plt.plot(mse_results['val_loss'], label='Mean Squared Error')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    
    # Log to wandb
    wandb.log({"loss_comparison": wandb.Image('loss_comparison.png')})
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 5))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(cross_entropy_results['accuracy'], label='Cross Entropy')
    plt.plot(mse_results['accuracy'], label='Mean Squared Error')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(cross_entropy_results['val_accuracy'], label='Cross Entropy')
    plt.plot(mse_results['val_accuracy'], label='Mean Squared Error')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    
    # Log to wandb
    wandb.log({"accuracy_comparison": wandb.Image('accuracy_comparison.png')})
