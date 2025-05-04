[![DOI](https://zenodo.org/badge/977500314.svg)](https://doi.org/10.5281/zenodo.15337322)
# UTScan: Urban Travel Intensity Estimation from Seismic Noise

UTScan is a hybrid machine learning framework that leverages continuous seismic noise recordings to quantify daily urban travel intensity. By combining CatBoost regression with an LSTM sequence model, it captures both nonlinear relationships and temporal dynamics for accurate, generalizable predictions across multiple cities and time periods.

---

## Features

* **Baseline Normalization**: Aligns noise levels across stations using low-activity baselines.
* **Two-stage Modeling**: CatBoost for nonlinear feature extraction; LSTM for temporal residual correction.
* **Flexible Configuration**: Easily switch between pure CatBoost, pure sequence, or fusion models.
* **Visualization**: Generate time-series plots comparing predicted vs. actual travel intensity.
* **Automatic DOI Generation**: Seamless integration with Zenodo for code archiving and DOI issuance.

---

## Requirements

* Python 3.7+
* Packages: `numpy`, `pandas`, `matplotlib`, `torch`, `catboost`, `openpyxl`, `scikit-learn`

Install via:

```bash
pip install -r requirements.txt
```

---

## Installation & Setup

1. **Clone the repository**:

   ```bash
   ```

git clone [https://github.com/](https://github.com/)<username>/UTScan.git
cd UTScan

```
2. **Configure paths and hyperparameters** in `config.py`. See the example below with English comments.
3. **Prepare data**:
- Place PSD text files in `./data/psd/`  
- Put travel intensity Excel in `./data/travel_intensity.xlsx`  
- Provide optional baseline file `./data/baseline_noise.txt`
4. **Push to GitHub** and create a Release (e.g., `v1.0.0`) to trigger DOI generation via Zenodo.

---

## Repository Structure

```

UTScan/
├── config.py             # Configuration (paths, dates, hyperparameters)
├── data\_loader.py        # Load and preprocess PSD & travel data
├── catboost\_module.py    # Train CatBoost with hyperparameter search
├── train\_utils.py        # Create sliding windows & train sequence models
├── model.py              # LSTM/Fusion model definitions
├── evaluator.py          # Compute evaluation metrics
├── main.py               # Multi-round training script
├── run\_predict.py        # Inference & plotting for independent data
├── outputs/              # Saved models, logs, and outputs
└── README.md             # This file

````

---

## Configuration (`config.py`)
```python
# config.py
import os

# Data file path configuration
DATA_DIR = "./data/psd/"              # PSD feature text file directory (including BHZ channel data)
TRAVEL_DATA_PATH = "./data/城内出行强度.xlsx"   # Excel file path for urban travel intensity data
MAPPING_PATH = "./data/城市-台站对应表.xlsx"    # Excel file path for city–station mapping table
LOW_VALUES_PATH = "./data/273个城市低值.txt"   # File path for baseline low noise levels (leave empty if not available)

# Data analysis start and end date (inclusive), format YYYY-MM-DD
START_DATE = "2020-01-01"
END_DATE   = "2020-04-30"

# Training / testing city split settings
TRAIN_CITY_NAMES = []   # List of city names designated for training (empty to use all)
TEST_CITY_NAMES  = []   # List of city names designated for testing (excluded from training)
TEST_SIZE        = 0.0  # If TEST_CITY_NAMES is empty, fraction of cities held out as test set (0 = no split)
RANDOM_SEED      = 0    # Seed for random splitting

# Time window length (days) for sequence model input
WINDOW_SIZE = 7

# CatBoost hyperparameter grid
CAT_PARAM_GRID = {
    "iterations": [2000],
    "learning_rate": [0.1, 0.01, 0.001],
    "depth": [3, 5, 7, 9],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0]
}

# Sequence model hyperparameter grid (LSTM/GRU/Transformer)
SEQ_PARAM_GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 3, 5],
    "batch_size": [16, 32, 64],
    "learning_rate": [0.1, 0.01],
    "activation": ["linear", "relu"]
}

# Model switches
USE_CATBOOST       = True      # Train/use CatBoost regression
USE_SEQUENCE_MODEL = True      # Train/use sequence model
MODEL_TYPE         = "LSTM"    # "LSTM", "GRU", or "Transformer"
USE_ATTENTION      = False     # Apply attention mechanism in sequence model

# Training parameters
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 5
MULTI_TRAIN_ROUNDS = 50

# Output paths
OUTPUT_DIR          = "outputs"
BEST_CAT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_catboost_model.cbm")
BEST_SEQ_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_sequence_model.pth")
```python
# config.py
import os

# Data file path configuration
DATA_DIR = "./data/psd/"              # PSD feature text file directory (including BHZ channel data)
TRAVEL_DATA_PATH = "./data/travel_intensity.xlsx"   # Excel file path for urban travel intensity data
MAPPING_PATH = "./data/city_station_map.xlsx"      # Excel file path for city–station mapping table
LOW_VALUES_PATH = "./data/baseline_noise.txt"      # File path for baseline low noise levels (leave empty if not available)

# Data analysis start and end date (inclusive), format YYYY-MM-DD
START_DATE = "2020-01-01"
END_DATE   = "2020-04-30"

# Training / testing city split settings
TRAIN_CITY_NAMES = []   # List of city names designated for training (empty to use all)
TEST_CITY_NAMES  = []   # List of city names designated for testing (excluded from training)
TEST_SIZE        = 0.0  # If TEST_CITY_NAMES is empty, fraction of cities held out as test set (0 = no split)
RANDOM_SEED      = 0    # Seed for random splitting

# Time window length (days) for sequence model input
WINDOW_SIZE = 7

# CatBoost hyperparameter grid
CAT_PARAM_GRID = {
    "iterations": [2000],
    "learning_rate": [0.1, 0.01, 0.001],
    "depth": [3, 5, 7, 9],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0]
}

# Sequence model hyperparameter grid (LSTM/GRU/Transformer)
SEQ_PARAM_GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 3, 5],
    "batch_size": [16, 32, 64],
    "learning_rate": [0.1, 0.01],
    "activation": ["linear", "relu"]
}

# Model switches
USE_CATBOOST       = True      # Train/use CatBoost regression
USE_SEQUENCE_MODEL = True      # Train/use sequence model
MODEL_TYPE         = "LSTM"   # "LSTM", "GRU", or "Transformer"
USE_ATTENTION      = False     # Apply attention mechanism in sequence model

# Training parameters
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 5
MULTI_TRAIN_ROUNDS = 50

# Output paths\ OUTPUT_DIR          = "outputs"
BEST_CAT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_catboost_model.cbm")
BEST_SEQ_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_sequence_model.pth")
````

---

## Data Loading (`data_loader.py`)

```python
# data_loader.py
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import logging
import config

def load_data():
    """
    Load and preprocess data for urban travel intensity prediction.
    Returns:
        city_names (list): Cities loaded
        X_seq_list (list of np.ndarray): Daily PSD feature arrays
        Y_seq_list (list of np.ndarray): Daily travel intensity arrays
    """
    # 1. Load baseline low-noise levels (optional)
    baseline_noise = {}
    if os.path.exists(config.LOW_VALUES_PATH):
        with open(config.LOW_VALUES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                code, vals = line.strip().split(maxsplit=1)
                baseline_noise[code] = eval(vals)

    # 2. Load city–station mapping
    df_map = pd.read_excel(config.MAPPING_PATH, dtype=str, engine='openpyxl', header=None)
    mapping = {row[0].strip(): row[1].strip() for _, row in df_map.iterrows()}

    # 3. Load travel intensity data
    df_travel = pd.read_excel(config.TRAVEL_DATA_PATH, engine='openpyxl', dtype=str)
    city_index = df_travel.columns[1]
    df_travel.set_index(city_index, inplace=True)
    df_travel.columns = df_travel.columns.map(str)

    # 4. Date range
    start = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(config.END_DATE, "%Y-%m-%d")
    date_range = [(start + timedelta(days=i)).strftime("%Y%m%d") 
                  for i in range((end-start).days+1)]

    # 5. Find PSD files
    psd_files = glob.glob(os.path.join(config.DATA_DIR, "*_BHZ_*.txt"))
    file_map = {os.path.basename(fp): fp for fp in psd_files}

    city_names, X_seq_list, Y_seq_list = [], [], []
    target_periods = [0.05, 0.058, 0.068, 0.079, 0.1, 0.11, 0.117, 0.126, 0.155, 0.2, 0.23, 0.317, 0.49]

    for station, city in mapping.items():
        # Construct filename e.g. "HB_WHN_BHZ_2020.txt"
        net, sta = station.split('.',1)
        fname = f"{net}_{sta}_BHZ_{start.year}.txt"
        if fname not in file_map:
            logging.warning(f"Missing PSD file: {fname}")
            continue
        psd_path = file_map[fname]

        # Align travel series
        travel = df_travel.loc[city].reindex(date_range).ffill().bfill().astype(float)

        # Parse PSD hourly, extract features, aggregate daily
        daily_hours = {}
        with open(psd_path,'r',encoding='utf-8',errors='ignore') as f:
            prev_ts = None
            pairs = []
            for raw in f:
                line = raw.strip()
                if not line: continue
                token = line.split()[0]
                # detect new hour marker
                if len(token)>=18 and token[15]=='_' and token[18]=='_':
                    # aggregate previous hour
                    if prev_ts and pairs:
                        feats=[]
                        for T in target_periods:
                            p=1.08*T
                            closest=min(pairs, key=lambda x:abs(x[0]-p))[1]
                            feats.append(closest)
                        day=prev_ts[:8]
                        daily_hours.setdefault(day,[]).append(feats)
                    # reset
                    prev_ts = token[11:19].replace('_','')
                    pairs=[]
                    data_str = raw.split(None,1)[1] if ' ' in raw else ''
                else:
                    data_str = line
                # parse period;noise segments
                if data_str:
                    for seg in data_str.split('&'):
                        if ';' not in seg: continue
                        prd,noise = seg.split(';',1)
                        try:
                            prd_val=float(prd)
                            noise_val=float(noise)
                        except: continue
                        if 0.05<=prd_val<=0.55:
                            pairs.append((prd_val,noise_val))
            # last hour
            if prev_ts and pairs:
                feats=[min(pairs,key=lambda x:abs(x[0]-1.08*T))[1] for T in target_periods]
                daily_hours.setdefault(prev_ts[:8],[]).append(feats)

        # Daily feature matrix
        df_feat = pd.DataFrame({d:np.mean(v,axis=0) 
                                 for d,v in daily_hours.items()}).T.reindex(date_range).ffill().bfill()
        # Baseline subtraction
        if station in baseline_noise:
            df_feat -= baseline_noise[station]

        city_names.append(city)
        X_seq_list.append(df_feat.values.astype(float))
        Y_seq_list.append(travel.values.astype(float))

    return city_names, X_seq_list, Y_seq_list
```

---

## Training & Evaluation

See `main.py` for multi-round random-split training (`--multi_train`), and `run_predict.py` for independent inference and plotting.
All evaluation functions are in `evaluator.py`, computing RMSE, MAE, MAPE, R², and Pearson correlation.

---

## GitHub & DOI Setup

1. **Create a Public repository** on GitHub and push this project (see instructions in `docs/GITHUB.md`).
2. **Enable GitHub–Zenodo integration**, create a Release tags (e.g., `v1.0.0`), then obtain a DOI automatically on Zenodo.

---

## Citation

```bibtex
@software{UTScan2025,
  author       = {Your Name},
  title        = {UTScan: Urban Travel Intensity from Seismic Noise},
  version      = {v1.0.0},
  year         = {2025},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/<username>/UTScan}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
