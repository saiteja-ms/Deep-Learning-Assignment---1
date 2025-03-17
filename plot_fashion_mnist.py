import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Initialize Wandb
wandb.init(project='Fashion_mnist_sweep_backprop', entity='teja_sai-indian-institute-of-technology-madras')

# Load Fashion-MNIST dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot one sample from each class
plt.figure(figsize=(10, 10))
for i in range(10):
    # Find the first instance of each class
    idx = np.where(y_train == i)[0][0]  # Fixed indexing issue
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_samples.png')

# Log the plot to Wandb
wandb.log({"fashion_mnist_samples": wandb.Image('fashion_mnist_samples.png')})

# Finish Wandb run
wandb.finish()
