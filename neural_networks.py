import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init # Store weight initialization method

        # Initialize weights based on specific method
        if weight_init == 'random':
            self.weights = np.random.randn(input_size, output_size)*0.01
        elif weight_init == 'Xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1/input_size)
        else:
            raise ValueError(f"Unsupported weight initialization method: {weight_init}")
        
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
        self.z = None # Store pre-activation values
        self.dweights = None
        self.dbiases = None
        self.delta = None # Store delta values for backprop

    def forward(self, input_data):
        self.input = input_data 
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self._apply_activation(self.z)
        return self.output
    
    def backward(self, delta, learning_rate):
        # Calculate gradient for current layer
        batch_size = delta.shape[0]

        # Gradient of weights
        self.dweights = np.dot(self.input.T, delta) / batch_size

        # Gradient of biases
        self.dbiases = np.sum(delta, axis=0, keepdims=True) / batch_size

        # Gradient to pass to previous layer
        if self.activation != 'softmax':
            delta_prev = np.dot(delta, self.weights.T) * self._activation_derivative(self.z)
        else:
            delta_prev = np.dot(delta, self.weights.T)
        
        return delta_prev
        
    def _apply_activation(self, z):
        # Apply activation function
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        
        elif self.activation == 'tanh':
            return np.tanh(z)
        
        elif self.activation == 'ReLU':
            return np.maximum(0, z)
        
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def _activation_derivative(self, z):
        # Get derivative of activation function
        if self.activation == 'sigmoid':
            sig = self._apply_activation(z)
            return sig * (1 - sig)
        
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        
        elif self.activation == 'ReLU':
            return np.where(z>0, 1, 0)
        
        elif self.activation == 'identity':
            return np.ones_like(z)
        
        else:
            raise ValueError(f"Derivative not implemented for activation: {self.activation}")

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='sigmoid', weight_init='random'):
        self.layers = []
        self.activation = activation # Store activation function
        self.weight_init = weight_init # Store weight init method

        # Input to first hidden layer
        self.layers.append(Layer(input_size, hidden_sizes[0], activation, weight_init))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], activation, weight_init))

        # Output layer - Use softmax for the last layer
        self.layers.append(Layer(hidden_sizes[i-1], output_size, 'softmax',  weight_init))

    def forward(self, X):
        # Forward pass through all layers
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def backward(self, y, learning_rate, optimizer, optimizer_params):
        # Backward pass with specified optimizer
        delta = y # Delta will be the gradient of the loss function for the last layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

            # Update layer weights and biases using the optimizer
            optimizer.update(layer)

    def train(self, X, y, epochs, batch_size, learning_rate, optimizer='sgd', **optimizer_params):
        # Training loop
        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Convert optimizer string to object
                if optimizer == 'sgd':
                    optim = SGD(learning_rate=learning_rate, weight_decay=optimizer_params.get('weight_decay', 0))
                elif optimizer == 'momentum':
                    optim = Momentum(learning_rate=learning_rate, momentum=optimizer_params.get('momentum', 0.9), weight_decay=optimizer_params.get('weight_decay', 0))
                elif optimizer == 'nag':
                    optim = NAG(learning_rate=learning_rate, momentum=optimizer_params.get('momentum', 0.9), weight_decay=optimizer_params.get('weight_decay', 0))
                elif optimizer == 'rmsprop':
                    optim = RMSprop(learning_rate=learning_rate, beta=optimizer_params.get('beta', 0.9), weight_decay=optimizer_params.get('weight_decay', 0))
                elif optimizer == 'adam':
                    optim = Adam(learning_rate=learning_rate, beta1=optimizer_params.get('beta1', 0.9), beta2=optimizer_params.get('beta2', 0.999), epsilon=optimizer_params.get('epsilon', 1e-8), weight_decay=optimizer_params.get('weight_decay', 0))
                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer}")
                
                # Backward pass
                if(optimizer == 'sgd'):
                    self.backward(y_batch, learning_rate, optim, optimizer_params)
                else:
                    self.backward(y_batch, learning_rate, optim, optimizer_params)

    def predict(self, X):
        # Make predictions
        return np.argmax(self.forward(X), axis=1)





