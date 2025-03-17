import numpy as np
import matplotlib.pyplot as plt
import wandb
from src.data import load_data
from src.model import NeuralNetwork

# Initialize wandb
wandb.init(project='Fashion_mnist_sweep_backprop', entity='teja_sai-indian-institute-of-technology-madras')

# Load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data('fashion_mnist')

# Create model with your best hyperparameters
input_size = X_train.shape[1]  # Corrected
hidden_sizes = [128, 128, 128]
output_size = 10

model = NeuralNetwork(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    output_size=output_size,
    activation="ReLU",
    weight_init="Xavier"
)

# Find one example from each class
class_examples = []
for class_idx in range(10):
    indices = np.where(y_test == class_idx)[0]
    if len(indices) > 0:
        class_examples.append(indices[0])

# Get predictions for one example from each class
X_samples = X_test[class_examples]
y_samples = y_test[class_examples]
y_pred = model.forward(X_samples)

# Ensure the output is a probability distribution
def is_probability_distribution(output):
    return np.allclose(np.sum(output, axis=1), 1) and np.all(output >= 0) and np.all(output <= 1)

assert is_probability_distribution(y_pred), "The output is not a valid probability distribution."

# Plot probability distributions for all classes
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(10):
    # Plot the probability distribution
    axes[i].bar(range(10), y_pred[i])
    axes[i].set_xticks(range(10))
    axes[i].set_xticklabels(range(10), rotation=45, fontsize=8)
    axes[i].set_title(f"Class: {class_names[y_samples[i]]}")

plt.tight_layout()
plt.savefig('all_class_probabilities.png')

# Also create a visualization showing images and their distributions
fig, axes = plt.subplots(10, 2, figsize=(10, 25))
for i in range(10):
    # Plot the image
    axes[i, 0].imshow(X_samples[i].reshape(28, 28), cmap='gray')
    axes[i, 0].set_title(f"Class: {class_names[y_samples[i]]}")
    axes[i, 0].axis('off')
    
    # Plot the probability distribution
    axes[i, 1].bar(range(10), y_pred[i])
    axes[i, 1].set_xticks(range(10))
    axes[i, 1].set_xticklabels(class_names, rotation=90, fontsize=6)
    axes[i, 1].set_title(f"Prediction: {class_names[np.argmax(y_pred[i])]}")

plt.tight_layout()
plt.savefig('class_probabilities_with_images.png')

# Log to wandb
wandb.log({
    "all_class_probabilities": wandb.Image('all_class_probabilities.png'),
    "class_probabilities_with_images": wandb.Image('class_probabilities_with_images.png')
})
wandb.finish()
