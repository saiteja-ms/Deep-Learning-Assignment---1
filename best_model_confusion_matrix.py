import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.data import load_data
from src.model import NeuralNetwork

# Initialize wandb
wandb.init(project='fashion_mnist_best_model', entity='teja_sai')

# Load test data
(_, _), (_, _), (X_test, y_test) = load_data('fashion_mnist')

# Create model with your best hyperparameters (replace with your best config)
input_size = X_test.shape[1]
hidden_sizes = [128, 128, 128]  # Example - use your best model's architecture
output_size = 10

model = NeuralNetwork(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    activation="ReLU",  # Use your best model's activation
    weight_init="Xavier"  # Use your best model's weight init
)

# Get predictions
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a more creative visualization
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create a more visually appealing confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='viridis', values_format='d')
plt.title('Confusion Matrix for Best Fashion-MNIST Model', fontsize=18, fontweight='bold')

# Add accuracy information
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
plt.figtext(0.5, 0.01, f'Test Accuracy: {accuracy:.4f}', 
            ha='center', fontsize=16, bbox={'facecolor':'lightgreen', 'alpha':0.5, 'pad':5})

# Save and log to wandb
plt.savefig('confusion_matrix.png')
wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png'),
           "test_accuracy": accuracy})

wandb.finish()
