import numpy as np
import wandb
from tqdm import tqdm
from src.activation import Activation
from src.loss import Loss
from src.optimizers import Optimizer

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU", weight_init="Xavier"):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        if weight_init == "random":
            self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
        elif weight_init == "Xavier":
            self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / (input_size + hidden_sizes[0])))
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            if weight_init == "random":
                self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 0.01)
            elif weight_init == "Xavier":
                self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 
                                   np.sqrt(2.0 / (hidden_sizes[i-1] + hidden_sizes[i])))
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        
        # Last hidden layer to output layer
        if weight_init == "random":
            self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
        elif weight_init == "Xavier":
            self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 
                               np.sqrt(2.0 / (hidden_sizes[-1] + output_size)))
        self.biases.append(np.zeros((1, output_size)))
        
        # Set activation functions
        self.activation, self.activation_derivative = Activation.get_activation(activation)
        
        # For softmax output
        self.softmax = Activation.softmax
    
    def forward(self, X):
        # Store activations and pre-activations for backpropagation
        self.z_values = []
        self.a_values = [X]  # Input layer activation
        
        # Forward pass through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.a_values[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activation(z)
            self.a_values.append(a)
        
        # Output layer (softmax activation for classification)
        z_out = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        a_out = self.softmax(z_out)
        self.a_values.append(a_out)
        
        return a_out
    
    def backward(self, X, y, y_pred, loss_derivative):
        batch_size = X.shape[0]
        
        # Convert y to one-hot encoding if needed
        if len(y.shape) == 1:
            y_one_hot = np.zeros((y.size, self.output_size))
            y_one_hot[np.arange(y.size), y] = 1
            y = y_one_hot
        
        # Initialize gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = loss_derivative(y, y_pred)
        
        # Backpropagate through layers
        for l in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for weights and biases
            dw[l] = np.dot(self.a_values[l].T, delta) / batch_size
            db[l] = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Backpropagate error to previous layer (if not input layer)
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(self.z_values[l-1])
        
        return dw, db
    
    def update_parameters(self, dw, db, optimizer, optimizer_config):
        # Combine weights and biases for optimizer
        params = self.weights + self.biases
        grads = dw + db
        
        # Apply optimizer
        updated_params, updated_config = optimizer(params, grads, optimizer_config)
        
        # Split updated parameters back into weights and biases
        self.weights = updated_params[:len(self.weights)]
        self.biases = updated_params[len(self.weights):]
        
        return updated_config
    
    def predict(self, X):
        # Forward pass
        y_pred = self.forward(X)
        
        # Return class with highest probability
        return np.argmax(y_pred, axis=1)
    
    def evaluate(self, X, y):
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y)
        
        return accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, loss_name, optimizer_name, optimizer_config, wandb_log=True):
        # Get loss function and its derivative
        loss_fn, loss_derivative = Loss.get_loss(loss_name)
        
        # Get optimizer
        optimizer = Optimizer.get_optimizer(optimizer_name)
        
        # Number of batches
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Initialize metrics for this epoch
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Mini-batch training
            for batch in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                # Get batch data
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                batch_loss = loss_fn(y_batch, y_pred)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
                
                # Compute accuracy
                batch_accuracy = np.mean(np.argmax(y_pred, axis=1) == y_batch)
                epoch_accuracy += batch_accuracy * (end_idx - start_idx) / n_samples
                
                # Backward pass
                dw, db = self.backward(X_batch, y_batch, y_pred, loss_derivative)
                
                # Update parameters
                optimizer_config = self.update_parameters(dw, db, optimizer, optimizer_config)
            
            # Evaluate on validation set
            val_pred = self.forward(X_val)
            val_loss = loss_fn(y_val, val_pred)
            val_accuracy = self.evaluate(X_val, y_val)
            
            # Update history
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Log to wandb
            if wandb_log:
                wandb.log({
                    'loss': epoch_loss,
                    'accuracy': epoch_accuracy,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'epoch': epoch
                })
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
        
        return history
