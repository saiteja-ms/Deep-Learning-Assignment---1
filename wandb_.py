import wandb
import numpy as np
from Loss_function import cross_entropy_loss, mean_squared_error, cross_entropy_gradient, mse_gradient

def train_with_wandb(model, X_train, y_train, X_val, y_val, config):
    # Initialize wandb run
    run = wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)

    # Training loop
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        # Mini-batch training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Compute loss
            if config['loss'] == 'cross_entropy':
                batch_loss = cross_entropy_loss(y_batch, y_pred)
                delta = cross_entropy_gradient(y_batch, y_pred)
            else:  # mean_squared_error
                batch_loss = mean_squared_error(y_batch, y_pred)
                delta = mse_gradient(y_batch, y_pred)
            
            epoch_loss += batch_loss
            
            # Backward pass
            # Pass config directly instead of calling optimizer_params
            model.backward(delta, config['learning_rate'], config['optimizer'], config)
        
        # Calculate validation metrics
        val_pred = model.forward(X_val)
        if config['loss'] == 'cross_entropy':
            val_loss = cross_entropy_loss(y_val, val_pred)
        else:
            val_loss = mean_squared_error(y_val, val_pred)
        
        # Evaluate accuracy using the predict function
        y_val_pred = model.predict(X_val)
        val_accuracy = np.mean(y_val_pred == np.argmax(y_val, axis=1))
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss / num_batches,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })

    # Close wandb run
    wandb.finish()

def optimizer_params(config):
    # Extract optimizer parameters from config
    optimizer_params = {}
    if config['optimizer'] in ['momentum', 'nag']:
        optimizer_params['momentum'] = config['momentum']
    elif config['optimizer'] == 'rmsprop':
        optimizer_params['beta'] = config['beta']
    elif config['optimizer'] in ['adam', 'nadam']:
        optimizer_params['beta1'] = config['beta1']
        optimizer_params['beta2'] = config['beta2']
        optimizer_params['epsilon'] = config['epsilon']
    return optimizer_params