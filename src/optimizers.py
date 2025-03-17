import numpy as np

class Optimizer:
    @staticmethod
    def sgd(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.01)
        weight_decay = config.get('weight_decay', 0.0)
        
        for param, grad in zip(params, grads):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Update parameters
            param -= learning_rate * grad
        
        return params, config
    
    @staticmethod
    def momentum(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)
        
        # Initialize velocity if not already done
        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Update velocity
            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grad
            
            # Update parameters
            param += config['velocity'][i]
        
        return params, config
    
    @staticmethod
    def nag(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)
        
        # Initialize velocity if not already done
        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Store old velocity
            old_velocity = config['velocity'][i].copy()
            
            # Update velocity
            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grad
            
            # Update parameters with Nesterov correction
            param += -momentum * old_velocity + (1 + momentum) * config['velocity'][i]
        
        return params, config
    
    @staticmethod
    def rmsprop(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.01)
        beta = config.get('beta', 0.9)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        # Initialize cache if not already done
        if 'cache' not in config:
            config['cache'] = [np.zeros_like(param) for param in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Update cache
            config['cache'][i] = beta * config['cache'][i] + (1 - beta) * grad**2
            
            # Update parameters
            param -= learning_rate * grad / (np.sqrt(config['cache'][i]) + epsilon)
        
        return params, config
    
    @staticmethod
    def adam(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.001)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        # Initialize m and v if not already done
        if 'm' not in config:
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]
            config['t'] = 0
        
        config['t'] += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Update biased first moment estimate
            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = config['m'][i] / (1 - beta1**config['t'])
            
            # Compute bias-corrected second raw moment estimate
            v_hat = config['v'][i] / (1 - beta2**config['t'])
            
            # Update parameters
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return params, config
    
    @staticmethod
    def nadam(params, grads, config=None):
        if config is None:
            config = {}
        
        learning_rate = config.get('learning_rate', 0.001)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        # Initialize m and v if not already done
        if 'm' not in config:
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]
            config['t'] = 0
        
        config['t'] += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # L2 regularization
            if weight_decay > 0:
                grad = grad + weight_decay * param
            
            # Update biased first moment estimate
            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grad**2
            
            # Compute bias-corrected first moment estimate
            m_hat = config['m'][i] / (1 - beta1**config['t'])
            
            # Compute bias-corrected second raw moment estimate
            v_hat = config['v'][i] / (1 - beta2**config['t'])
            
            # Nesterov accelerated gradient term
            m_hat_nesterov = (beta1 * m_hat + (1 - beta1) * grad) / (1 - beta1**config['t'])
            
            # Update parameters
            param -= learning_rate * m_hat_nesterov / (np.sqrt(v_hat) + epsilon)
        
        return params, config
    
    @staticmethod
    def get_optimizer(name):
        if name == "sgd":
            return Optimizer.sgd
        elif name == "momentum":
            return Optimizer.momentum
        elif name == "nag":
            return Optimizer.nag
        elif name == "rmsprop":
            return Optimizer.rmsprop
        elif name == "adam":
            return Optimizer.adam
        elif name == "nadam":
            return Optimizer.nadam
        else:
            raise ValueError(f"Optimizer {name} not supported")
