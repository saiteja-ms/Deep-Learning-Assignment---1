import numpy as np

class Optimizer:
    """
    Class containing optimization algorithms.
    """
    @staticmethod
    def sgd(params, grads, config):
        """
        Stochastic Gradient Descent optimizer.
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        weight_decay = config.get('weight_decay', 0.0)
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            params[i] = param - learning_rate * grads[i]
        
        return params, config
    
    @staticmethod
    def momentum(params, grads, config):
        """
        Momentum optimizer.
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)
        
        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            
            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grads[i]
            params[i] = param + config['velocity'][i]
        
        return params, config
    
    @staticmethod
    def nag(params, grads, config):
        """
        Nesterov Accelerated Gradient optimizer.
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        momentum = config.get('momentum', 0.9)
        weight_decay = config.get('weight_decay', 0.0)
        
        if 'velocity' not in config:
            config['velocity'] = [np.zeros_like(param) for param in params]
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            
            v_prev = config['velocity'][i].copy()
            config['velocity'][i] = momentum * config['velocity'][i] - learning_rate * grads[i]
            params[i] = param - momentum * v_prev + (1 + momentum) * config['velocity'][i]
        
        return params, config
    
    @staticmethod
    def rmsprop(params, grads, config):
        """
        RMSprop optimizer.
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        beta = config.get('beta', 0.9)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        if 'cache' not in config:
            config['cache'] = [np.zeros_like(param) for param in params]
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            
            config['cache'][i] = beta * config['cache'][i] + (1 - beta) * grads[i]**2
            params[i] = param - learning_rate * grads[i] / (np.sqrt(config['cache'][i]) + epsilon)
        
        return params, config
    
    @staticmethod
    def adam(params, grads, config):
        """
        Adam optimizer.
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        if 't' not in config:
            config['t'] = 0
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]
        
        config['t'] += 1
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            
            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grads[i]
            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grads[i]**2
            
            m_hat = config['m'][i] / (1 - beta1**config['t'])
            v_hat = config['v'][i] / (1 - beta2**config['t'])
            
            params[i] = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return params, config
    
    @staticmethod
    def nadam(params, grads, config):
        """
        Nadam optimizer (Adam with Nesterov momentum).
        
        Args:
            params (list): List of parameters to update
            grads (list): List of gradients
            config (dict): Optimizer configuration
            
        Returns:
            tuple: (updated_params, updated_config)
        """
        learning_rate = config.get('learning_rate', 0.01)
        beta1 = config.get('beta1', 0.9)
        beta2 = config.get('beta2', 0.999)
        epsilon = config.get('epsilon', 1e-8)
        weight_decay = config.get('weight_decay', 0.0)
        
        if 't' not in config:
            config['t'] = 0
            config['m'] = [np.zeros_like(param) for param in params]
            config['v'] = [np.zeros_like(param) for param in params]
        
        config['t'] += 1
        
        for i, param in enumerate(params):
            if weight_decay > 0:
                grads[i] = grads[i] + weight_decay * param
            
            config['m'][i] = beta1 * config['m'][i] + (1 - beta1) * grads[i]
            config['v'][i] = beta2 * config['v'][i] + (1 - beta2) * grads[i]**2
            
            m_hat = config['m'][i] / (1 - beta1**config['t'])
            v_hat = config['v'][i] / (1 - beta2**config['t'])
            
            m_hat_next = beta1 * m_hat + (1 - beta1) * grads[i] / (1 - beta1**config['t'])
            
            params[i] = param - learning_rate * m_hat_next / (np.sqrt(v_hat) + epsilon)
        
        return params, config
    
    @staticmethod
    def get_optimizer(optimizer_name):
        """
        Get optimizer function by name.
        
        Args:
            optimizer_name (str): Name of optimizer
            
        Returns:
            function: Optimizer function
        """
        if optimizer_name == "sgd":
            return Optimizer.sgd
        elif optimizer_name == "momentum":
            return Optimizer.momentum
        elif optimizer_name == "nag":
            return Optimizer.nag
        elif optimizer_name == "rmsprop":
            return Optimizer.rmsprop
        elif optimizer_name == "adam":
            return Optimizer.adam
        elif optimizer_name == "nadam":
            return Optimizer.nadam
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
