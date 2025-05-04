import os
import numpy as np
import logging
import torch
import random
import csv
import config
from data_loader import load_data
from catboost_module import train_catboost
from train_utils import create_sequences, train_sequence_model
from evaluator import compute_metrics, compute_metrics2
from model import get_model
import argparse
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor

def compute_additional_metrics(y_true, y_pred):
    """
    Compute MAE, MAPE, and Pearson correlation coefficient
    """
    mae = mean_absolute_error(y_true, y_pred)
    if np.all(y_true != 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mape = 0.0
    if len(y_true) > 1:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        corr = 0.0
    return mae, mape, corr


def multi_train():
    """
    Multi-round training mode:
      - Use 2020-01-01 to 2020-01-20 as the test set
      - Randomly split the period 01-21 to END into 70% train / 30% validation
      - Grid search for CatBoost and LSTM
      - Output RMSE, R2, MAE, MAPE, Pearson r
      - Write to log and CSV
      - Save the best models
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Logging file configuration
    log_path = os.path.join(config.OUTPUT_DIR, "multi_train.log")
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    # CSV output
    csv_path = os.path.join(config.OUTPUT_DIR, "multi_train_results.csv")
    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "run",
            "Cat_val_RMSE", "Cat_val_R2", "Cat_val_MAE", "Cat_val_MAPE", "Cat_val_r",
            "Cat_test_RMSE", "Cat_test_R2", "Cat_test_MAE", "Cat_test_MAPE", "Cat_test_r",
            "Seq_val_RMSE", "Seq_val_R2", "Seq_val_MAE", "Seq_val_MAPE", "Seq_val_r"
        ])

        # Initialize result storage
        cat_models = []
        seq_models = []
        seq_metrics = []  # store (rmse, r2)

        # Construct CatBoost parameter grid
        cb_keys = list(config.CAT_PARAM_GRID.keys())
        cb_combos = [
            dict(zip(cb_keys, vals))
            for vals in product(*(config.CAT_PARAM_GRID[k] for k in cb_keys))
        ]
        total_cb = len(cb_combos)

        # Construct LSTM parameter grid
        seq_keys = list(config.SEQ_PARAM_GRID.keys())
        seq_combos = [
            dict(zip(seq_keys, vals))
            for vals in product(*(config.SEQ_PARAM_GRID[k] for k in seq_keys))
        ]
        total_seq = len(seq_combos)

        test_len = 20  # First 20 days as test set

        for run in range(1, config.MULTI_TRAIN_ROUNDS + 1):
            msg = f"[Run {run}/{config.MULTI_TRAIN_ROUNDS}] start"
            print(msg, flush=True)
            logging.info(msg)

            # 1. Load data
            city_names, X_seq_list, Y_seq_list = load_data()
            logging.info(f"Loaded data for {len(city_names)} cities")

            # 2. Construct CatBoost datasets
            X_tr_cb, y_tr_cb = [], []
            X_val_cb, y_val_cb = [], []
            X_te_cb, y_te_cb = [], []
            for X_seq, Y_seq in zip(X_seq_list, Y_seq_list):
                X_te_cb.append(X_seq[:test_len]); y_te_cb.append(Y_seq[:test_len])
                rem_X, rem_y = X_seq[test_len:], Y_seq[test_len:]
                if len(rem_y) < 2:
                    continue
                idx = np.random.permutation(len(rem_y))
                split = int(0.7 * len(idx))
                tr_i, val_i = idx[:split], idx[split:]
                X_tr_cb.append(rem_X[tr_i]); y_tr_cb.append(rem_y[tr_i])
                X_val_cb.append(rem_X[val_i]); y_val_cb.append(rem_y[val_i])

            X_tr_cb = np.vstack(X_tr_cb)
            y_tr_cb = np.concatenate(y_tr_cb)
            X_val_cb = np.vstack(X_val_cb)
            y_val_cb = np.concatenate(y_val_cb)
            X_te_cb = np.vstack(X_te_cb)
            y_te_cb = np.concatenate(y_te_cb)

            # 3. CatBoost grid search
            best_cb = None
            best_cb_params = None
            best_val_rmse_cb = float('inf')
            best_val_r2_cb = best_val_mae_cb = best_val_mape_cb = best_val_corr_cb = None

            for idx_cb, params in enumerate(cb_combos, start=1):
                print(f"  [CatBoost {idx_cb}/{total_cb}] params={params}", flush=True)
                logging.info(f"CatBoost combo {idx_cb}/{total_cb}: {params}")
                p = params.copy()
                p.setdefault('loss_function', 'RMSE')
                p.setdefault('random_seed', 0)
                p.setdefault('verbose', False)
                model = CatBoostRegressor(**p)
                model.fit(
                    X_tr_cb, y_tr_cb,
                    eval_set=(X_val_cb, y_val_cb),
                    early_stopping_rounds=50,
                    use_best_model=True,
                    verbose=False
                )
                pred_val = model.predict(X_val_cb)
                rmse = np.sqrt(mean_squared_error(y_val_cb, pred_val))
                r2   = r2_score(y_val_cb, pred_val)
                mae, mape, corr = compute_additional_metrics(y_val_cb, pred_val)
                logging.info(f"    -> Val RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, r={corr:.4f}")
                if rmse < best_val_rmse_cb:
                    best_val_rmse_cb, best_val_r2_cb = rmse, r2
                    best_val_mae_cb, best_val_mape_cb, best_val_corr_cb = mae, mape, corr
                    best_cb, best_cb_params = model, p

            # CatBoost test set evaluation
            pred_te_cb = best_cb.predict(X_te_cb)
            te_rmse_cb = np.sqrt(mean_squared_error(y_te_cb, pred_te_cb))
            te_r2_cb   = r2_score(y_te_cb, pred_te_cb)
            te_mae_cb, te_mape_cb, te_corr_cb = compute_additional_metrics(y_te_cb, pred_te_cb)
            logging.info(
                f"Run {run} Best CB params={best_cb_params!s}, "
                f"Val RMSE={best_val_rmse_cb:.4f}, R2={best_val_r2_cb:.4f}, MAE={best_val_mae_cb:.4f}, MAPE={best_val_mape_cb:.2f}%, r={best_val_corr_cb:.4f}; "
                f"Test RMSE={te_rmse_cb:.4f}, R2={te_r2_cb:.4f}, MAE={te_mae_cb:.4f}, MAPE={te_mape_cb:.2f}%, r={te_corr_cb:.4f}"
            )
            print(f"  [CB Best] ValRMSE={best_val_rmse_cb:.4f}, TestRMSE={te_rmse_cb:.4f}", flush=True)

            # 4. Construct LSTM input features
            logging.info("Constructing LSTM features")
            X_feat_seq, Y_win_seq = [], []
            for X_seq, Y_seq in zip(X_seq_list, Y_seq_list):
                rem_X, rem_y = X_seq[test_len:], Y_seq[test_len:]
                if len(rem_y) <= config.WINDOW_SIZE:
                    continue
                preds = best_cb.predict(rem_X).reshape(-1, 1)
                feat = np.hstack([rem_X, preds]) if config.MODEL_TYPE == "FusionB" else preds
                Xw, Yw = create_sequences(feat, rem_y, config.WINDOW_SIZE)
                if Xw.size:
                    X_feat_seq.append(Xw); Y_win_seq.append(Yw)

            X_feat_seq = np.vstack(X_feat_seq)
            Y_win_seq  = np.concatenate(Y_win_seq)
            idx_seq = np.random.permutation(len(Y_win_seq))
            split = int(0.7 * len(idx_seq))
            tr_i, val_i = idx_seq[:split], idx_seq[split:]
            X_tr_seq, Y_tr_seq = X_feat_seq[tr_i], Y_win_seq[tr_i]
            X_val_seq, Y_val_seq = X_feat_seq[val_i], Y_win_seq[val_i]

            # 5. LSTM grid search
            best_seq = None
            best_val_seq_rmse = float('inf')
            best_val_seq_r2 = best_val_seq_mae = best_val_seq_mape = best_val_seq_corr = None
            best_seq_hp = None

            print(f"  [LSTM Grid] total {total_seq} combinations", flush=True)
            for idx_seq_combo, hp in enumerate(seq_combos, start=1):
                print(f"    [LSTM {idx_seq_combo}/{total_seq}] hp={hp}", flush=True)
                logging.info(f"LSTM combo {idx_seq_combo}/{total_seq}: {hp}")
                seq_model, val_rmse = train_sequence_model(
                    X_tr_seq, Y_tr_seq,
                    X_val_seq, Y_val_seq,
                    hp["hidden_size"], hp["num_layers"],
                    hp["batch_size"], hp["learning_rate"],
                    model_type=config.MODEL_TYPE,
                    use_attention=config.USE_ATTENTION,
                    activation=hp["activation"],
                    num_epochs=config.NUM_EPOCHS,
                    patience=config.EARLY_STOPPING_PATIENCE
                )
                logging.info(f"    -> Val RMSE={val_rmse:.4f}")
                if val_rmse < best_val_seq_rmse:
                    best_val_seq_rmse = val_rmse
                    best_seq = seq_model
                    best_seq_hp = hp
                    # Compute other metrics
                    device = next(best_seq.parameters()).device
                    best_seq.eval()
                    inp = torch.from_numpy(X_val_seq).float().to(device)
                    with torch.no_grad():
                        pred_seq = best_seq(inp).cpu().numpy().flatten()
                    best_val_seq_r2 = r2_score(Y_val_seq, pred_seq)
                    best_val_seq_mae, best_val_seq_mape, best_val_seq_corr = compute_additional_metrics(Y_val_seq, pred_seq)
                    logging.info(f"    ** Best R2={best_val_seq_r2:.4f}, MAE={best_val_seq_mae:.4f}, MAPE={best_val_seq_mape:.2f}%, r={best_val_seq_corr:.4f}")

            logging.info(
                f"Run {run} Best LSTM hp={best_seq_hp!s}, "
                f"Val RMSE={best_val_seq_rmse:.4f}, R2={best_val_seq_r2:.4f}, "
                f"MAE={best_val_seq_mae:.4f}, MAPE={best_val_seq_mape:.2f}%, r={best_val_seq_corr:.4f}"
            )
            print(f"  [LSTM Best] ValRMSE={best_val_seq_rmse:.4f}, ValR2={best_val_seq_r2:.4f}", flush=True)

            # Store this run's models and metrics
            cat_models.append(best_cb)
            seq_models.append(best_seq)
            seq_metrics.append((best_val_seq_rmse, best_val_seq_r2))

            # Write to CSV
            writer.writerow([
                run,
                f"{best_val_rmse_cb:.4f}", f"{best_val_r2_cb:.4f}", f"{best_val_mae_cb:.4f}", f"{best_val_mape_cb:.2f}%", f"{best_val_corr_cb:.4f}",
                f"{te_rmse_cb:.4f}", f"{te_r2_cb:.4f}", f"{te_mae_cb:.4f}", f"{te_mape_cb:.2f}%", f"{te_corr_cb:.4f}",
                f"{best_val_seq_rmse:.4f}", f"{best_val_seq_r2:.4f}", f"{best_val_seq_mae:.4f}", f"{best_val_seq_mape:.2f}%", f"{best_val_seq_corr:.4f}"
            ])

        # After multi-round training
        print("===== Multi-round training complete =====", flush=True)

        # Select best run based on Seq validation RMSE
        seq_vals = [m[0] for m in seq_metrics]
        best_run = int(np.argmin(seq_vals)) + 1
        best_msg = (
            f"Best run: {best_run}, "
            f"CatBoost params={cat_models[best_run-1].get_params()}, "
            f"LSTM best hyperparams={best_seq_hp!s}, "
            f"Seq ValRMSE={seq_metrics[best_run-1][0]:.4f}"
        )
        print(best_msg); logging.info(best_msg)

        # Save best models
        cat_models[best_run-1].save_model(
            os.path.join(config.OUTPUT_DIR, "best_multi_cb.cbm")
        )
        torch.save(
            seq_models[best_run-1],
            os.path.join(config.OUTPUT_DIR, "best_multi_seq.pth")
        )
        logging.info("Saved best models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi_train",
        action="store_true",
        help="Enable 200-round CatBoost+LSTM random training evaluation"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if args.multi_train:
        multi_train()
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(os.path.join(config.OUTPUT_DIR, "train.log"), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    results_path = os.path.join(config.OUTPUT_DIR, "results.csv")
    out_file = open(results_path, "w", newline='')
    writer = csv.writer(out_file)
    writer.writerow(["Model", "iterations", "learning_rate", "depth", "l2_leaf_reg", "bagging_temperature",
                     "hidden_size", "num_layers", "batch_size", "activation", "Val_RMSE", "Val_MAE", "Val_R2", "Val_MAPE", "Val_Pearson"])
    logging.info("Starting to load data...")
    city_names, X_seq_list, Y_seq_list = load_data()
    logging.info(f"Data loaded: {len(city_names)} cities")

    # If using fusion model, enable both CatBoost and sequence model options
    if hasattr(config, "MODEL_TYPE") and config.MODEL_TYPE in ["FusionA", "FusionB"]:
        config.USE_CATBOOST = True
        config.USE_SEQUENCE_MODEL = True

    # Determine train and test city index lists
    train_idx = []
    test_idx = []
    if config.TEST_CITY_NAMES:
        # Split based on specified test cities
        for i, city in enumerate(city_names):
            if city in config.TEST_CITY_NAMES:
                test_idx.append(i)
            else:
                train_idx.append(i)
    elif config.TEST_SIZE and config.TEST_SIZE > 0:
        # Randomly split a portion of cities as test set
        total_indices = list(range(len(city_names)))
        random.seed(config.RANDOM_SEED)
        random.shuffle(total_indices)
        test_count = int(config.TEST_SIZE * len(city_names))
        test_idx = total_indices[:test_count]
        train_idx = total_indices[test_count:]
        train_idx.sort()
        test_idx.sort()
    else:
        # No test city split, use all cities for training
        train_idx = list(range(len(city_names)))
        test_idx = []

    # Prepare containers for CatBoost and sequence model train/validation data
    X_train_cb_all, y_train_cb_all = [], []
    X_val_cb_all, y_val_cb_all = [], []
    train_seq_data, val_seq_data = [], []
    # Collect baseline model validation and test results (baseline uses mean of training period)
    baseline_pred_val_all = np.array([])
    baseline_true_val_all = np.array([])
    baseline_pred_test_all = np.array([])
    baseline_true_test_all = np.array([])

    # Iterate each training city to split time series segments
    for i in train_idx:
        X_seq = X_seq_list[i]
        Y_seq = Y_seq_list[i]
        n = len(Y_seq)
        # Split 70% train, 30% validation
        train_cut = int(0.7 * n)
        val_cut = int(1.0 * n)
        if val_cut == n:
            val_cut = n - 1  # ensure at least one day for validation

        # Collect CatBoost training and validation data
        X_train_cb_all.append(X_seq[:train_cut])
        y_train_cb_all.append(Y_seq[:train_cut])
        X_val_cb_all.append(X_seq[train_cut:val_cut])
        y_val_cb_all.append(Y_seq[train_cut:val_cut])

        # If not using CatBoost, prepare sliding-window data for sequence model
        if config.USE_SEQUENCE_MODEL and not config.USE_CATBOOST:
            X_win_train, Y_win_train = create_sequences(X_seq[:train_cut], Y_seq[:train_cut], config.WINDOW_SIZE)
            X_win_val, Y_win_val = create_sequences(X_seq[train_cut:val_cut], Y_seq[train_cut:val_cut], config.WINDOW_SIZE)
            if X_win_train.size > 0:
                train_seq_data.append((X_win_train, Y_win_train))
            if X_win_val.size > 0:
                val_seq_data.append((X_win_val, Y_win_val))

    # Combine CatBoost training/validation data
    X_train_cb = np.vstack(X_train_cb_all) if X_train_cb_all else np.empty((0,))
    y_train_cb = np.concatenate(y_train_cb_all) if y_train_cb_all else np.empty((0,))
    X_val_cb = np.vstack(X_val_cb_all) if X_val_cb_all else np.empty((0,))
    y_val_cb = np.concatenate(y_val_cb_all) if y_val_cb_all else np.empty((0,))

    # Train CatBoost model
    best_cat_model = None
    best_cat_params = None
    cat_results = None
    if config.USE_CATBOOST:
        logging.info("Starting CatBoost training and hyperparameter search...")
        best_cat_model, best_cat_params, cat_results = train_catboost(X_train_cb, y_train_cb, X_val_cb, y_val_cb, config.CAT_PARAM_GRID)
        if best_cat_model:
            best_cat_model.save_model(config.BEST_CAT_MODEL_PATH)
            logging.info(f"Best CatBoost model saved to {config.BEST_CAT_MODEL_PATH}")
        if cat_results:
            for params, rmse, mae, r2, mape, corr in cat_results:
                writer.writerow([
                    "CatBoost", params.get("iterations", ""), params.get("learning_rate", ""), params.get("depth", ""), 
                    params.get("l2_leaf_reg", ""), params.get("bagging_temperature", ""),
                    "", "", "", "",
                    f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}", f"{mape:.2f}", f"{corr:.4f}"
                ])

    # Train sequence model (LSTM/GRU/Transformer or Fusion)
    best_seq_model = None
    best_seq_params = None
    if config.USE_SEQUENCE_MODEL:
        # If CatBoost was used, use its predictions to build sequence model input
        if config.USE_CATBOOST:
            train_seq_data = []
            val_seq_data = []
            for i in train_idx:
                X_seq = X_seq_list[i]
                Y_seq = Y_seq_list[i]
                n = len(Y_seq)
                train_cut = int(0.7 * n)
                val_cut = int(1.0 * n)
                if val_cut == n:
                    val_cut = n - 1
                pred_full = best_cat_model.predict(X_seq) if best_cat_model else np.zeros(n)
                if config.MODEL_TYPE == "FusionB":
                    X_full_seq = np.hstack([X_seq, pred_full.reshape(-1, 1)])
                else:
                    X_full_seq = pred_full.reshape(-1, 1)
                X_win_train, Y_win_train = create_sequences(X_full_seq[:train_cut], Y_seq[:train_cut], config.WINDOW_SIZE)
                X_win_val, Y_win_val = create_sequences(X_full_seq[train_cut:val_cut], Y_seq[train_cut:val_cut], config.WINDOW_SIZE)
                if X_win_train.size > 0:
                    train_seq_data.append((X_win_train, Y_win_train))
                if X_win_val.size > 0:
                    val_seq_data.append((X_win_val, Y_win_val))

        # Combine all cities' sequence train/validation data
        if train_seq_data:
            X_train_seq_all = np.vstack([d[0] for d in train_seq_data])
            Y_train_seq_all = np.concatenate([d[1] for d in train_seq_data])
        else:
            feat_dim = (X_seq_list[0].shape[1] + 1) if (config.USE_CATBOOST and config.MODEL_TYPE == "FusionB") else (1 if config.USE_CATBOOST else X_seq_list[0].shape[1])
            X_train_seq_all = np.empty((0, config.WINDOW_SIZE, feat_dim))
            Y_train_seq_all = np.array([])
        if val_seq_data:
            X_val_seq_all = np.vstack([d[0] for d in val_seq_data])
            Y_val_seq_all = np.concatenate([d[1] for d in val_seq_data])
        else:
            feat_dim = (X_seq_list[0].shape[1] + 1) if (config.USE_CATBOOST and config.MODEL_TYPE == "FusionB") else (1 if config.USE_CATBOOST else X_seq_list[0].shape[1])
            X_val_seq_all = np.empty((0, config.WINDOW_SIZE, feat_dim))
            Y_val_seq_all = np.array([])

        logging.info("Starting sequence model training and hyperparameter search...")
        best_val_rmse = float('inf')
        for activation in config.SEQ_PARAM_GRID.get("activation", ["linear"]):
            for hidden_size in config.SEQ_PARAM_GRID.get("hidden_size", [32]):
                for num_layers in config.SEQ_PARAM_GRID.get("num_layers", [1]):
                    for batch_size in config.SEQ_PARAM_GRID.get("batch_size", [32]):
                        for lr in config.SEQ_PARAM_GRID.get("learning_rate", [0.001]):
                            logging.info(f"Training {config.MODEL_TYPE} (attention={config.USE_ATTENTION}) hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}")
                            if config.MODEL_TYPE in ["FusionA", "FusionB"]:
                                config.HIDDEN_SIZE = hidden_size
                                config.NUM_LAYERS = num_layers
                                config.BATCH_SIZE = batch_size
                                config.LEARNING_RATE = lr
                                config.ACTIVATION = activation
                                model = get_model(config)
                                model.fit(X_train_seq_all, Y_train_seq_all, X_val_seq_all, Y_val_seq_all)
                                val_pred = model.predict(X_val_seq_all)
                                val_pred = np.array(val_pred).flatten()
                            else:
                                model, _ = train_sequence_model(
                                    X_train_seq_all, Y_train_seq_all, 
                                    X_val_seq_all, Y_val_seq_all,
                                    hidden_size, num_layers, batch_size, lr,
                                    model_type=config.MODEL_TYPE, use_attention=config.USE_ATTENTION, activation=activation,
                                    num_epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE
                                )
                                model.eval()
                                with torch.no_grad():
                                    val_tensor = torch.from_numpy(X_val_seq_all).float().to(next(model.parameters()).device)
                                    val_pred = model(val_tensor).cpu().numpy().flatten()
                            metrics_val = compute_metrics2(Y_val_seq_all, val_pred)
                            logging.info(f"Params: hidden_size={hidden_size}, num_layers={num_layers}, batch_size={batch_size}, learning_rate={lr}, activation={activation} -> Val RMSE: {metrics_val['RMSE']:.4f}, MAE: {metrics_val['MAE']:.4f}, R2: {metrics_val['R2']:.4f}, MAPE: {metrics_val['MAPE']:.2f}%, Pearson: {metrics_val['Pearson']:.4f}")
                            writer.writerow([
                                config.MODEL_TYPE, "", "", "", "", "",
                                hidden_size, num_layers, batch_size, activation,
                                f"{metrics_val['RMSE']:.4f}", f"{metrics_val['MAE']:.4f}", f"{metrics_val['R2']:.4f}", f"{metrics_val['MAPE']:.2f}", f"{metrics_val['Pearson']:.4f}"
                            ])
                            if metrics_val["RMSE"] < best_val_rmse:
                                best_val_rmse = metrics_val["RMSE"]
                                best_seq_model = model
                                best_seq_params = {"hidden_size": hidden_size, "num_layers": num_layers, "batch_size": batch_size,
                                                   "learning_rate": lr, "activation": activation}
        if best_seq_model:
            torch.save(best_seq_model, config.BEST_SEQ_MODEL_PATH)
            logging.info(f"Best sequence model saved to {config.BEST_SEQ_MODEL_PATH}")

    # Notify to use separate evaluation module for test set evaluation
    logging.info("Training complete. For independent dataset evaluation, please run run_predict.py")
    if 'out_file' in locals():
        out_file.close()
    logging.info("Training process and results have been recorded in the log file and results.csv")

if __name__ == "__main__":
    main()
