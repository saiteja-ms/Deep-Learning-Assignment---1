import numpy as np
class Optimizer:
    def __init__(self, learning_rate=0.01, weight_decay=0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer):
        # Update layer weights and biases
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0):
        super().__init__(learning_rate, weight_decay)

    def update(self, layer):
        # Apply weight decay if specified
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_w = {}
        self.velocity_b = {}

    def update(self, layer):
        # Initialize velocity if not exist
        if id(layer) not in self.velocity_w:
            self.velocity_w[id(layer)] = np.zeros_like(layer.weights)
            self.velocity_b[id(layer)] = np.zeros_like(layer.biases)

        # Apply weight decay
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        # Update velocity and parameters
        self.velocity_w[id(layer)] = self.momentum * self.velocity_w[id(layer)] - self.learning_rate * layer.dweights
        self.velocity_b[id(layer)] = self.momentum * self.velocity_b[id(layer)] - self.learning_rate * layer.dbiases

        layer.weights -= self.velocity_w[id(layer)]
        layer.biases -= self.velocity_b[id(layer)]

class NAG(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_w = {}
        self.velocity_b = {}

    def update(self, layer):
        # Initialize velocity if not exist
        if id(layer) not in self.velocity_w:
            self.velocity_w[id(layer)] = np.zeros_like(layer.weights)
            self.velocity_b[id(layer)] = np.zeros_like(layer.biases)

        # Store previous velocities
        if id(layer) not in self.velocity_w:
          prev_velocity_w = np.zeros_like(layer.weights)
          prev_velocity_b = np.zeros_like(layer.biases)
        else:
          prev_velocity_w = self.velocity_w[id(layer)].copy()
          prev_velocity_b = self.velocity_b[id(layer)].copy()

        # Apply weight decay
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        # Update velocity
        self.velocity_w[id(layer)] = self.momentum * self.velocity_w[id(layer)] - self.learning_rate * layer.dweights
        self.velocity_b[id(layer)] = self.momentum * self.velocity_b[id(layer)] - self.learning_rate * layer.dbiases

        # Update parameters with Nesterov look-ahead
        layer.weights -= self.momentum * prev_velocity_w + (1 + self.momentum) * self.velocity_w[id(layer)]
        layer.biases -= self.momentum * prev_velocity_b + (1 + self.momentum) * self.velocity_b[id(layer)]

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.square_grad_w = {}
        self.square_grad_b = {}

    def update(self, layer):
        # Initialize accumulated squared gradients if not exist
        if id(layer) not in self.square_grad_w:
            self.square_grad_w[id(layer)] = np.zeros_like(layer.weights)
            self.square_grad_b[id(layer)] = np.zeros_like(layer.biases)

        # Apply weight decay
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        # Update accumulated squared gradients
        self.square_grad_w[id(layer)] = self.beta * self.square_grad_w[id(layer)] + (1 - self.beta) * np.square(layer.dweights)
        self.square_grad_b[id(layer)] = self.beta * self.square_grad_b[id(layer)] + (1 - self.beta) * np.square(layer.dbiases)

        # Update parameters
        layer.weights -= (self.learning_rate / (np.sqrt(self.square_grad_w[id(layer)] + self.epsilon))) * layer.dweights
        layer.biases -= (self.learning_rate / (np.sqrt(self.square_grad_b[id(layer)] + self.epsilon))) * layer.dbiases

class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
        self.t = 0 # Time step

    def update(self, layer):
        self.t += 1

        # Initialize first and second moment variables
        if id(layer) not in self.m_w:
            self.m_w[id(layer)] = np.zeros_like(layer.weights)
            self.m_b[id(layer)] = np.zeros_like(layer.biases)
            self.v_w[id(layer)] = np.zeros_like(layer.weights)
            self.v_b[id(layer)] = np.zeros_like(layer.biases)

        # Apply weight decay
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        # Update biased first moment estimate and second moment estimates
        m_w_hat = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[id(layer)] / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[id(layer)] / (1 - self.beta2 ** self.t)

        # Update parameters
        layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # Update first and second moment
        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.dweights
        self.m_b[id(layer)] = self.beta1 * self.m_b[id(layer)] + (1 - self.beta1) * layer.dbiases
        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * np.square(layer.dweights)
        self.v_b[id(layer)] = self.beta2 * self.v_b[id(layer)] + (1 - self.beta2) * np.square(layer.dbiases)

class NAdam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
        self.t = 0 # Time step

    def update(self, layer):
        self.t += 1

        # Initialize biased moments if not exist
        if id(layer) not in self.m_w:
            self.m_w[id(layer)] = np.zeros_like(layer.weights)
            self.m_b[id(layer)] = np.zeros_like(layer.biases)
            self.v_w[id(layer)] = np.zeros_like(layer.weights)
            self.v_b[id(layer)] = np.zeros_like(layer.biases)

        # Apply weight decay
        if self.weight_decay > 0:
            layer.dweights += self.weight_decay * layer.weights

        # Update biased first moment estimate and second moment estimates
        self.m_w[id(layer)] = self.beta1 * self.m_w[id(layer)] + (1 - self.beta1) * layer.dweights
        self.m_b[id(layer)] = self.beta1 * self.m_b[id(layer)] + (1 - self.beta1) * layer.dbiases
        self.v_w[id(layer)] = self.beta2 * self.v_w[id(layer)] + (1 - self.beta2) * np.square(layer.dweights)
        self.v_b[id(layer)] = self.beta2 * self.v_b[id(layer)] + (1 - self.beta2) * np.square(layer.dbiases)

        # Compute bias-corrected first and second moment estimates
        m_w_hat = self.m_w[id(layer)] / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b[id(layer)] / (1 - self.beta1 ** self.t)

        v_w_hat = self.v_w[id(layer)] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[id(layer)] / (1 - self.beta2 ** self.t)

        # Incorporate Nesterov momentum into the Adam update
        m_w_nesterov = self.beta1 * m_w_hat + (1 - self.beta1) * layer.dweights / (1 - self.beta1 ** self.t)
        m_b_nesterov = self.beta1 * m_b_hat + (1 - self.beta1) * layer.dbiases / (1 - self.beta1 ** self.t)

        # Update parameters with NAdam
        layer.weights -= self.learning_rate * m_w_nesterov / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= self.learning_rate * m_b_nesterov / (np.sqrt(v_b_hat) + self.epsilon)