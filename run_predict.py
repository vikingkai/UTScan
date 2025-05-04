# run_predict.py
import sys
# Hijack config module to config_test.py
import config_test as config
sys.modules['config'] = config

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging

from data_loader import load_data
from catboost import CatBoostRegressor
from evaluator import compute_metrics2
# scaler = np.load(os.path.join(config.OUTPUT_DIR, "scaler.npz"))
# feat_mean = scaler['mean']
# feat_std  = scaler['std']

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # 2) Load CatBoost & sequence model & scaler
    cat_model = CatBoostRegressor()
    cat_model.load_model(config.BEST_CAT_MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_model = torch.load(config.BEST_SEQ_MODEL_PATH, map_location=device)
    seq_model.eval()
    # scaler = np.load(os.path.join(config.OUTPUT_DIR, "scaler.npz"))
    # feat_mean = scaler['mean']
    # feat_std  = scaler['std']

    # 3) Load and preprocess 2019 data
    logging.info("Loading 2019 PSD features and travel intensity data...")
    city_names, X_seq_list, Y_seq_list = load_data()

    if not city_names:
        raise RuntimeError("No 2019 data loaded; please check configuration")

    all_dates = pd.date_range(config.START_DATE, config.END_DATE).to_pydatetime().tolist()

    # 7) Predict and save
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    all_true, all_pred_cat, all_pred_final = [], [], []
    for idx, city in enumerate(city_names):
        X_seq = X_seq_list[idx]
        Y_seq = Y_seq_list[idx] if idx < len(Y_seq_list) else None

        # 7.1 CatBoost prediction
        y_pred_cat = cat_model.predict(X_seq).flatten()
        logging.info(f"{city} CatBoost sample preds: {y_pred_cat[:5]}")

        # 7.2 Sequence model correction
        if config.USE_SEQUENCE_MODEL:
            seq_preds = []
            for t in range(len(y_pred_cat)):
                if t < config.WINDOW_SIZE - 1:
                    seq_preds.append(y_pred_cat[t])
                else:
                    window = y_pred_cat[t-config.WINDOW_SIZE+1:t+1]
                    inp = torch.from_numpy(window.reshape(1, -1, 1)).float().to(device)
                    with torch.no_grad():
                        seq_preds.append(seq_model(inp).item())
            y_pred_final = np.array(seq_preds)
        else:
            y_pred_final = y_pred_cat

        # 7.3 Compute overall metrics
        if Y_seq is not None:
            metrics = compute_metrics2(Y_seq, y_pred_final)
            logging.info(f"{city} 2019 Test - RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}, Pearson={metrics['Pearson']:.4f}")

        # 7.4 Construct DataFrame and save
        dates = all_dates[:len(y_pred_final)]
        df_out = pd.DataFrame({
            "date": dates,
            "predicted": y_pred_final
        })
        if Y_seq is not None:
            df_out["actual"] = Y_seq
            all_true.extend(Y_seq.tolist())
            all_pred_cat.extend(y_pred_cat.tolist())
            all_pred_final.extend(y_pred_final.tolist())

        fn = f"{city.replace('市','')}_2019_pred.csv"
        df_out.to_csv(os.path.join(config.OUTPUT_DIR, fn), index=False, encoding='utf-8')

        # 7.5 Plotting
        plt.figure(figsize=(6, 3), dpi=300)
        if "actual" in df_out:
            plt.plot(df_out["date"], df_out["actual"], label="Actual")
        plt.plot(df_out["date"], df_out["predicted"], linestyle='--', label="Predicted")
        plt.title(f"{city} Travel Intensity (2019)")
        plt.xlabel("Date")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, fn.replace('.csv', '.png')))
        plt.close()

    # 8) Global summary metrics
    if all_true:
        m_all = compute_metrics2(np.array(all_true), np.array(all_pred_final))
        logging.info(f"Overall 2019 fusion model - RMSE={m_all['RMSE']:.4f}, R2={m_all['R2']:.4f}, Pearson={m_all['Pearson']:.4f}, MAPE={m_all['MAPE']:.2f}%")

if __name__ == "__main__":
    main()
