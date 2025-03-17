import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
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
        idx = np.where(y_train == i)[0][0]
        
        # Plot
        plt.subplot(3, 4, i+1)
        plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png')
    plt.close()
    
    # Log to wandb
    wandb.log({"fashion_mnist_samples": wandb.Image('fashion_mnist_samples.png')})

def plot_confusion_matrix(model, X_test, y_test, dataset):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        model (NeuralNetwork): Trained neural network model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
        dataset (str): Dataset name ('fashion_mnist' or 'mnist')
        
    Returns:
        float: Test accuracy
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Class names
    if dataset == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:  # mnist
        class_names = [str(i) for i in range(10)]
    
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
    accuracy = accuracy_score(y_test, y_pred)
    plt.figtext(0.5, 0.01, f'Test Accuracy: {accuracy:.4f}', 
                ha='center', fontsize=16, bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':5})
    
    # Highlight the diagonal (correct predictions)
    for i in range(len(class_names)):
        plt.text(i+0.5, i+0.5, f'{cm[i,i]}', 
                ha='center', va='center', fontsize=15, color='white', 
                bbox={'facecolor':'green', 'alpha':0.6, 'pad':10})
    
    # Find the most confused classes
    np.fill_diagonal(cm, 0)  # Remove diagonal for finding max confusion
    max_confusion = np.unravel_index(np.argmax(cm), cm.shape)
    plt.text(max_confusion[1]+0.5, max_confusion[0]+0.5, f'{cm[max_confusion]}', 
            ha='center', va='center', fontsize=15, color='white', 
            bbox={'facecolor':'red', 'alpha':0.6, 'pad':10})
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Log to wandb
    wandb.log({
        "confusion_matrix": wandb.Image('confusion_matrix.png'),
        "test_accuracy": accuracy
    })
    
    return accuracy

def plot_loss_comparison(results):
    """
    Plot comparison of loss functions.
    
    Args:
        results (dict): Dictionary containing training history for different loss functions
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for loss_name, data in results.items():
        plt.plot(data['history']['loss'], label=loss_name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    for loss_name, data in results.items():
        plt.plot(data['history']['val_loss'], label=loss_name)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    plt.close()
    
    # Log to wandb
    wandb.log({"loss_comparison": wandb.Image('loss_comparison.png')})
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 5))
    
    # Plot training accuracy
    plt.subplot(1, 2, 1)
    for loss_name, data in results.items():
        plt.plot(data['history']['accuracy'], label=loss_name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for loss_name, data in results.items():
        plt.plot(data['history']['val_accuracy'], label=loss_name)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    # Log to wandb
    wandb.log({"accuracy_comparison": wandb.Image('accuracy_comparison.png')})
    
    # Print test accuracies
    for loss_name, data in results.items():
        print(f"Test Accuracy with {loss_name}: {data['test_accuracy']:.4f}")
        wandb.log({f"test_accuracy_{loss_name}": data['test_accuracy']})
