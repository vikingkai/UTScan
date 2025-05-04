# model.py

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostRegressor

# Evaluation metrics function
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute RMSE, MAE, MAPE, R2, Pearson correlation between actual and predicted arrays."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    # RMSE
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    # MAPE (percent, avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_pred - y_true)[mask] / y_true[mask])) * 100 if mask.any() else np.inf
    # R2 Score
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    # Pearson correlation
    if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        pearson = 0.0
    else:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Pearson": pearson}

# Sequence data preparation for FusionA
def create_sequences_fusionA(actual: np.ndarray, baseline: np.ndarray, seq_len: int, len_train: int):
    """
    Create sliding window sequences for FusionA (CatBoost + LSTM residual model).
    Inputs:
      actual   - Full actual value series for train+val (concatenated).
      baseline - Full CatBoost predicted series for train+val (concatenated).
      seq_len  - Number of past days to use for each sequence.
      len_train - Length of training set (to separate train/val sequences).
    Outputs:
      train_seqs, train_tgts, val_seqs, val_tgts – NumPy arrays for LSTM input sequences and targets.
        (val_seqs, val_tgts will be None if no validation data provided)
    """
    seq_list = []
    tgt_list = []
    idx_list = []
    len_total = len(actual)
    # Slide window from index seq_len to end of series
    for t in range(seq_len, len_total):
        # Input: past seq_len days of [CatBoost_pred, Actual] values
        seq_input = np.column_stack((baseline[t-seq_len:t], actual[t-seq_len:t]))
        # Target: residual = actual(t) - baseline_pred(t)
        target = actual[t] - baseline[t]
        seq_list.append(seq_input)
        tgt_list.append(target)
        idx_list.append(t)
    seq_array = np.array(seq_list, dtype=np.float32)
    tgt_array = np.array(tgt_list, dtype=np.float32)
    idx_array = np.array(idx_list)
    # Split into train and val sets based on target index
    train_mask = idx_array < len_train
    val_mask = idx_array >= len_train
    train_seqs = seq_array[train_mask]
    train_tgts = tgt_array[train_mask]
    val_seqs = seq_array[val_mask] if val_mask.any() else None
    val_tgts = tgt_array[val_mask] if val_mask.any() else None
    return train_seqs, train_tgts, val_seqs, val_tgts

# Sequence data preparation for FusionB
def create_sequences_fusionB(actual: np.ndarray, baseline: np.ndarray, seq_len: int, len_train: int):
    """
    Create sliding window sequences for FusionB (CatBoost + LSTM direct fusion model).
    Inputs and outputs are similar to create_sequences_fusionA, but here sequences contain only CatBoost predictions.
    Target is the actual value (no residual).
    """
    seq_list = []
    tgt_list = []
    idx_list = []
    len_total = len(actual)
    for t in range(seq_len, len_total):
        # Input: past seq_len days of CatBoost predicted values (reshaped to (seq_len, 1))
        seq_input = baseline[t-seq_len:t].reshape(-1, 1)
        target = actual[t]  # Target: actual value directly
        seq_list.append(seq_input)
        tgt_list.append(target)
        idx_list.append(t)
    seq_array = np.array(seq_list, dtype=np.float32)
    tgt_array = np.array(tgt_list, dtype=np.float32)
    idx_array = np.array(idx_list)
    train_mask = idx_array < len_train
    val_mask = idx_array >= len_train
    train_seqs = seq_array[train_mask]
    train_tgts = tgt_array[train_mask]
    val_seqs = seq_array[val_mask] if val_mask.any() else None
    val_tgts = tgt_array[val_mask] if val_mask.any() else None
    return train_seqs, train_tgts, val_seqs, val_tgts

# LSTM neural network model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, activation: str):
        """
        LSTM model for time series forecasting.
        input_size: number of features in each time step (e.g., 2 for FusionA, 1 for FusionB).
        hidden_size: number of LSTM hidden units.
        num_layers: number of LSTM layers.
        activation: activation function for output ("linear" or "relu").
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # final fully connected layer to output prediction
        self.activation_type = activation.lower()
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                   # LSTM output shape: (batch, seq_len, hidden_size)
        last_out = out[:, -1, :]               # Take output of last time step for each sequence
        out = self.fc(last_out)                # Linear layer to get final output value
        if self.activation_type == "relu":
            out = torch.relu(out)              # apply ReLU activation if specified
        return out

class FusionA:
    """FusionA: CatBoost + LSTM residual prediction model."""
    def __init__(self, config):
        # Configuration parameters
        self.config = config
        self.seq_len = getattr(config, "seq_len", 10)  # number of past days in sequence (default 10 if not specified)
        # Initialize CatBoostRegressor with parameters from config
        cat_params = {
            "iterations": getattr(config, "catboost_iterations", 1000),
            "learning_rate": getattr(config, "catboost_learning_rate", 0.1),
            "depth": getattr(config, "catboost_depth", 6),
            "loss_function": getattr(config, "catboost_loss_function", "RMSE"),
            "verbose": False
        }
        self.cat_model = CatBoostRegressor(**cat_params)
        self.lstm_model = None    # will hold the trained LSTM model
        self.best_params = None   # best hyperparameters for LSTM found
        # Save historical context (last seq_len values) for prediction phase
        self.history_actual = None
        self.history_baseline = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the CatBoost model and LSTM residual model.
        X_train, y_train: training feature matrix and target array.
        X_val, y_val: optional validation feature matrix and target array for early stopping and hyperparameter tuning.
        """
        # Convert targets to NumPy arrays (float32)
        y_train = np.array(y_train, dtype=np.float32).ravel()
        y_val = np.array(y_val, dtype=np.float32).ravel() if y_val is not None else None

        # 1. Train CatBoost model on training data (with optional early stopping on validation set)
        if X_val is not None and y_val is not None and len(y_val) > 0:
            self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                               early_stopping_rounds=getattr(self.config, "catboost_es_rounds", 50),
                               use_best_model=True, verbose=False)
        else:
            self.cat_model.fit(X_train, y_train, verbose=False)

        # 2. Generate baseline predictions (CatBoost) for training and validation sets
        baseline_train = self.cat_model.predict(X_train)
        baseline_val = self.cat_model.predict(X_val) if X_val is not None else None

        # 3. Prepare combined series of actual and baseline predictions for sequence generation
        if baseline_val is not None and y_val is not None:
            actual_series = np.concatenate([y_train, y_val])      # actual values: train + val
            baseline_series = np.concatenate([baseline_train, baseline_val])  # baseline predictions: train + val
        else:
            actual_series = y_train
            baseline_series = baseline_train
        len_train = len(y_train)

        # 4. Create sequence datasets for LSTM (FusionA uses [baseline, actual] as inputs, residual as target)
        train_seqs, train_tgts, val_seqs, val_tgts = create_sequences_fusionA(actual_series, baseline_series, self.seq_len, len_train)

        # Save last seq_len points of series as history for future predictions
        self.history_actual = actual_series[-self.seq_len:]
        self.history_baseline = baseline_series[-self.seq_len:]

        # 5. Hyperparameter search for LSTM (if multiple values provided in config)
        # Define hyperparameter grid from config (or use single values if only one provided)
        hidden_sizes = self.config.hidden_size if isinstance(self.config.hidden_size, list) else [self.config.hidden_size]
        num_layers_list = self.config.num_layers if isinstance(self.config.num_layers, list) else [self.config.num_layers]
        batch_sizes = self.config.batch_size if isinstance(self.config.batch_size, list) else [self.config.batch_size]
        learning_rates = self.config.learning_rate if isinstance(self.config.learning_rate, list) else [self.config.learning_rate]
        activations = self.config.activation if isinstance(self.config.activation, list) else [self.config.activation]

        # If no validation set provided, restrict to a single combination (no reliable way to choose best without val)
        param_grid = []
        if val_seqs is not None and val_tgts is not None:
            # Full grid search combinations
            for hs in hidden_sizes:
                for nl in num_layers_list:
                    for bs in batch_sizes:
                        for lr in learning_rates:
                            for act in activations:
                                param_grid.append((hs, nl, bs, lr, act))
        else:
            # Use first (or only) values as the chosen hyperparameters
            param_grid.append((hidden_sizes[0], num_layers_list[0], batch_sizes[0], learning_rates[0], activations[0]))

        best_val_loss = float("inf")
        best_metrics = None
        best_params = None
        best_state_dict = None

        # Device configuration for training (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 6. Iterate over hyperparameter combinations
        for (hs, nl, bs, lr, act) in param_grid:
            hs, nl, bs = int(hs), int(nl), int(bs)
            lr = float(lr)
            act = str(act).lower()
            # Initialize LSTM model with current hyperparameters
            model = LSTMModel(input_size=2, hidden_size=hs, num_layers=nl, activation=act).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            # Create DataLoader for training sequences
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_seqs), 
                                                           torch.from_numpy(train_tgts).unsqueeze(1))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            # Early stopping settings
            patience = getattr(self.config, "early_stop_patience", 10)
            best_epoch_val_loss = float("inf")
            epochs_no_improve = 0
            num_epochs = getattr(self.config, "epochs", 100)

            # Training loop for LSTM
            for epoch in range(1, num_epochs + 1):
                model.train()
                total_loss = 0.0
                # Iterate over training batches
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(xb)
                train_loss = total_loss / len(train_dataset)
                # Compute validation loss for this epoch (if validation data exists)
                val_loss = None
                if val_seqs is not None and val_tgts is not None:
                    model.eval()
                    with torch.no_grad():
                        vb = torch.from_numpy(val_seqs).to(device)
                        vpred = model(vb).cpu().numpy().flatten()
                    val_loss = np.mean((vpred - val_tgts) ** 2)
                # Log the training/validation loss for this epoch
                if val_loss is not None:
                    print(f"[hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}] Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")
                else:
                    print(f"[hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}] Epoch {epoch}: TrainLoss={train_loss:.4f}")
                # Early stopping check on validation loss
                if val_loss is not None:
                    if val_loss < best_epoch_val_loss:
                        best_epoch_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"EarlyStopping triggered at epoch {epoch} for params (hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act})")
                        break

            # Evaluate metrics on validation set (or training set if no val) after training
            model.eval()
            if val_seqs is not None and val_tgts is not None:
                with torch.no_grad():
                    vpred = model(torch.from_numpy(val_seqs).to(device)).cpu().numpy().flatten()
                y_true = val_tgts
            else:
                with torch.no_grad():
                    tpred = model(torch.from_numpy(train_seqs).to(device)).cpu().numpy().flatten()
                y_true = train_tgts
                vpred = tpred  # in absence of val, use train predictions for metrics
            metrics = calc_metrics(y_true, vpred)
            print(f"Params(hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}) -> "
                  f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, "
                  f"R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}")

            # Check if this combination is the best so far (using validation loss if available, else training loss)
            current_val_loss = np.mean((vpred - y_true) ** 2)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_params = {"hidden_size": hs, "num_layers": nl, "batch_size": bs, "learning_rate": lr, "activation": act}
                best_metrics = metrics
                # Save model state (on CPU) for best combo
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        # 7. Save the best LSTM model and parameters
        if best_params is not None:
            # Re-initialize model with best parameters and load trained weights
            bp = best_params
            self.lstm_model = LSTMModel(input_size=2, hidden_size=bp["hidden_size"], 
                                        num_layers=bp["num_layers"], activation=bp["activation"])
            if best_state_dict is not None:
                self.lstm_model.load_state_dict(best_state_dict)
            self.lstm_model.eval()  # set to evaluation mode
            self.best_params = best_params
            # Log the best hyperparameter combination and its metrics
            print(f"Best LSTM hyperparameters: {best_params}")
            if best_metrics:
                print("Best validation metrics: "
                      f"RMSE={best_metrics['RMSE']:.4f}, MAE={best_metrics['MAE']:.4f}, "
                      f"MAPE={best_metrics['MAPE']:.2f}%, R2={best_metrics['R2']:.4f}, "
                      f"Pearson={best_metrics['Pearson']:.4f}")
        else:
            # In case no training happened (should not occur in normal use)
            self.lstm_model = None
            self.best_params = None

    def predict(self, X):
        """
        Generate predictions using the trained FusionA model.
        X: feature matrix for the period to predict (immediately after training/validation period).
        Returns a NumPy array of final predictions.
        """
        # 1. Get CatBoost baseline predictions for X (vectorized)
        baseline_pred = np.array(self.cat_model.predict(X), dtype=np.float32)
        if self.lstm_model is None:
            # If LSTM model is not trained (fallback to baseline predictions)
            return baseline_pred
        # 2. Use iterative approach to apply LSTM residual corrections
        preds = []
        # Prepare context lists from saved history (last seq_len known values)
        actual_context = list(self.history_actual.astype(np.float32))
        baseline_context = list(self.history_baseline.astype(np.float32))
        # Iteratively predict each time step in X
        for i in range(len(baseline_pred)):
            b_pred = baseline_pred[i]
            # Construct input sequence array of shape (1, seq_len, 2)
            seq_input = np.column_stack((baseline_context[-self.seq_len:], actual_context[-self.seq_len:])).astype(np.float32)
            seq_tensor = torch.from_numpy(seq_input).unsqueeze(0)  # shape: (1, seq_len, 2)
            # LSTM predicts the residual for this step
            with torch.no_grad():
                resid_pred = self.lstm_model(seq_tensor).item()
            # Final prediction = baseline + residual
            final_pred = b_pred + resid_pred
            preds.append(final_pred)
            # Update contexts for next step
            baseline_context.append(b_pred)
            # If actual value is not available (as in forecasting), use predicted value as proxy for actual
            actual_context.append(final_pred)
        return np.array(preds, dtype=np.float32)

class FusionB:
    """FusionB: CatBoost + LSTM direct fusion model."""
    def __init__(self, config):
        self.config = config
        self.seq_len = getattr(config, "seq_len", 10)
        cat_params = {
            "iterations": getattr(config, "catboost_iterations", 1000),
            "learning_rate": getattr(config, "catboost_learning_rate", 0.1),
            "depth": getattr(config, "catboost_depth", 6),
            "loss_function": getattr(config, "catboost_loss_function", "RMSE"),
            "verbose": False
        }
        self.cat_model = CatBoostRegressor(**cat_params)
        self.lstm_model = None
        self.best_params = None
        self.history_baseline = None  # last seq_len baseline values for context

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the CatBoost model and LSTM direct fusion model.
        """
        y_train = np.array(y_train, dtype=np.float32).ravel()
        y_val = np.array(y_val, dtype=np.float32).ravel() if y_val is not None else None

        # 1. Train CatBoost model
        if X_val is not None and y_val is not None and len(y_val) > 0:
            self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val),
                               early_stopping_rounds=getattr(self.config, "catboost_es_rounds", 50),
                               use_best_model=True, verbose=False)
        else:
            self.cat_model.fit(X_train, y_train, verbose=False)

        # 2. Generate baseline predictions for train and val sets
        baseline_train = self.cat_model.predict(X_train)
        baseline_val = self.cat_model.predict(X_val) if X_val is not None else None

        # 3. Combine actual and baseline series for sequence creation
        if baseline_val is not None and y_val is not None:
            actual_series = np.concatenate([y_train, y_val])
            baseline_series = np.concatenate([baseline_train, baseline_val])
        else:
            actual_series = y_train
            baseline_series = baseline_train
        len_train = len(y_train)

        # 4. Create sequence datasets for LSTM (FusionB uses baseline sequence as input, actual as target)
        train_seqs, train_tgts, val_seqs, val_tgts = create_sequences_fusionB(actual_series, baseline_series, self.seq_len, len_train)

        # Save last seq_len baseline predictions for context in prediction phase
        self.history_baseline = baseline_series[-self.seq_len:]

        # 5. Hyperparameter search (similar to FusionA)
        hidden_sizes = self.config.hidden_size if isinstance(self.config.hidden_size, list) else [self.config.hidden_size]
        num_layers_list = self.config.num_layers if isinstance(self.config.num_layers, list) else [self.config.num_layers]
        batch_sizes = self.config.batch_size if isinstance(self.config.batch_size, list) else [self.config.batch_size]
        learning_rates = self.config.learning_rate if isinstance(self.config.learning_rate, list) else [self.config.learning_rate]
        activations = self.config.activation if isinstance(self.config.activation, list) else [self.config.activation]

        param_grid = []
        if val_seqs is not None and val_tgts is not None:
            for hs in hidden_sizes:
                for nl in num_layers_list:
                    for bs in batch_sizes:
                        for lr in learning_rates:
                            for act in activations:
                                param_grid.append((hs, nl, bs, lr, act))
        else:
            param_grid.append((hidden_sizes[0], num_layers_list[0], batch_sizes[0], learning_rates[0], activations[0]))

        best_val_loss = float("inf")
        best_metrics = None
        best_params = None
        best_state_dict = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 6. Train and evaluate LSTM for each hyperparameter combination
        for (hs, nl, bs, lr, act) in param_grid:
            hs, nl, bs = int(hs), int(nl), int(bs)
            lr = float(lr)
            act = str(act).lower()
            model = LSTMModel(input_size=1, hidden_size=hs, num_layers=nl, activation=act).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_seqs), 
                                                           torch.from_numpy(train_tgts).unsqueeze(1))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
            patience = getattr(self.config, "early_stop_patience", 10)
            best_epoch_val_loss = float("inf")
            epochs_no_improve = 0
            num_epochs = getattr(self.config, "epochs", 100)
            for epoch in range(1, num_epochs + 1):
                model.train()
                total_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(xb)
                train_loss = total_loss / len(train_dataset)
                val_loss = None
                if val_seqs is not None and val_tgts is not None:
                    model.eval()
                    with torch.no_grad():
                        vb = torch.from_numpy(val_seqs).to(device)
                        vpred = model(vb).cpu().numpy().flatten()
                    val_loss = np.mean((vpred - val_tgts) ** 2)
                if val_loss is not None:
                    print(f"[hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}] Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")
                else:
                    print(f"[hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}] Epoch {epoch}: TrainLoss={train_loss:.4f}")
                if val_loss is not None:
                    if val_loss < best_epoch_val_loss:
                        best_epoch_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"EarlyStopping triggered at epoch {epoch} for params (hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act})")
                        break

            # Evaluate this model on validation (or training) data for metrics
            model.eval()
            if val_seqs is not None and val_tgts is not None:
                with torch.no_grad():
                    vpred = model(torch.from_numpy(val_seqs).to(device)).cpu().numpy().flatten()
                y_true = val_tgts
            else:
                with torch.no_grad():
                    tpred = model(torch.from_numpy(train_seqs).to(device)).cpu().numpy().flatten()
                y_true = train_tgts
                vpred = tpred
            metrics = calc_metrics(y_true, vpred)
            print(f"Params(hs={hs}, nl={nl}, bs={bs}, lr={lr}, act={act}) -> "
                  f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, "
                  f"R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}")
            current_val_loss = np.mean((vpred - y_true) ** 2)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_params = {"hidden_size": hs, "num_layers": nl, "batch_size": bs, "learning_rate": lr, "activation": act}
                best_metrics = metrics
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        # 7. Save best model and hyperparameters
        if best_params is not None:
            bp = best_params
            self.lstm_model = LSTMModel(input_size=1, hidden_size=bp["hidden_size"], 
                                        num_layers=bp["num_layers"], activation=bp["activation"])
            if best_state_dict is not None:
                self.lstm_model.load_state_dict(best_state_dict)
            self.lstm_model.eval()
            self.best_params = best_params
            print(f"Best LSTM hyperparameters: {best_params}")
            if best_metrics:
                print("Best validation metrics: "
                      f"RMSE={best_metrics['RMSE']:.4f}, MAE={best_metrics['MAE']:.4f}, "
                      f"MAPE={best_metrics['MAPE']:.2f}%, R2={best_metrics['R2']:.4f}, "
                      f"Pearson={best_metrics['Pearson']:.4f}")
        else:
            self.lstm_model = None
            self.best_params = None

    def predict(self, X):
        """
        Generate predictions using the trained FusionB model.
        """
        baseline_pred = np.array(self.cat_model.predict(X), dtype=np.float32)
        if self.lstm_model is None:
            return baseline_pred  # fallback to baseline if LSTM not available
        preds = []
        baseline_context = list(self.history_baseline.astype(np.float32))
        # Slide through each time step in X
        for i in range(len(baseline_pred)):
            b_pred = baseline_pred[i]
            # Prepare input sequence of baseline predictions (shape: (1, seq_len, 1))
            seq_input = np.array(baseline_context[-self.seq_len:], dtype=np.float32).reshape(1, self.seq_len, 1)
            seq_tensor = torch.from_numpy(seq_input)
            # LSTM directly outputs final prediction
            with torch.no_grad():
                final_pred = self.lstm_model(seq_tensor).item()
            preds.append(final_pred)
            # Append the baseline prediction to context for next step
            baseline_context.append(b_pred)
        return np.array(preds, dtype=np.float32)

def get_model(config):
    """Factory function to get model instance based on config.model_type."""
    mtype = str(config.model_type).lower()
    if mtype == "fusion_a":
        return FusionA(config)
    elif mtype == "fusion_b":
        return FusionB(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
SequenceModel = LSTMModel
class SequenceModel(LSTMModel):
    """
    兼容旧版 train_utils.py 的接口：
      SequenceModel(input_size, hidden_size, num_layers,
                    model_type=..., use_attention=..., activation=...)
    实际只是一个 LSTMModel，不用动 model_type/use_attention 参数。
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 model_type: str = 'LSTM',
                 use_attention: bool = False,
                 activation: str = 'linear'):
        # 忽略 model_type、use_attention，直接调用 LSTMModel 的构造
        super().__init__(input_size, hidden_size, num_layers, activation)

# 保留别名以兼容其他引用
SequenceModel = SequenceModel
