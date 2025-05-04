# train_utils.py
# Utility functions for creating sequence samples and training sequence models (LSTM)

import numpy as np
import torch
import logging
from model import SequenceModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_sequences(X_seq, Y_seq, window_size):
    """
    Convert feature sequence X_seq and target sequence Y_seq into supervised learning samples
    using a fixed-length sliding window.
    Returns:
        X_windows: shape (num_samples, window_size, num_features)
        Y_targets: shape (num_samples,)
    """
    X_seq = np.asarray(X_seq)
    Y_seq = np.asarray(Y_seq)
    X_windows = []
    Y_targets = []
    n = len(Y_seq)
    for i in range(n - window_size + 1):
        X_windows.append(X_seq[i:i+window_size])
        Y_targets.append(Y_seq[i+window_size-1])
    X_windows = np.array(X_windows)
    Y_targets = np.array(Y_targets)
    return X_windows, Y_targets

def train_sequence_model(X_train_seq, Y_train_seq, X_val_seq, Y_val_seq,
                         hidden_size, num_layers, batch_size, learning_rate,
                         model_type='LSTM', use_attention=False, activation='linear',
                         num_epochs=100, patience=5):
    """
    Train a sequence model (default LSTM) with the given hyperparameters.
    Returns the trained model and the best validation RMSE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequenceModel(X_train_seq.shape[2], hidden_size, num_layers,
                          model_type=model_type, use_attention=use_attention, activation=activation).to(device)
    criterion = torch.nn.MSELoss()  # MSE loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        # Shuffle training sequences each epoch
        perm = np.random.permutation(len(X_train_seq))
        X_train_seq = X_train_seq[perm]
        Y_train_seq = Y_train_seq[perm]
        # Batch training
        batch_count = int(np.ceil(len(X_train_seq) / batch_size))
        for b in range(batch_count):
            start = b * batch_size
            end = min(start + batch_size, len(X_train_seq))
            X_batch = torch.from_numpy(X_train_seq[start:end]).float().to(device)
            Y_batch = torch.from_numpy(Y_train_seq[start:end]).float().to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.flatten(), Y_batch)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val_seq).float().to(device)
            Y_val_tensor = torch.from_numpy(Y_val_seq).float().to(device)
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output.flatten(), Y_val_tensor).item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best model parameters from validation
    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_rmse = np.sqrt(best_val_loss) if best_val_loss != float('inf') else float('inf')
    return model, best_val_rmse
