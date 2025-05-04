# config.py
import os

# Data file path configuration
DATA_DIR = "./data/psd/"              # PSD feature text file directory (including BHZ channel data)
TRAVEL_DATA_PATH = "./data/travel_intensity.xlsx"   # Excel file path for urban travel intensity data
MAPPING_PATH = "./data/city_station.xlsx"    # Excel file path for city–station mapping table
LOW_VALUES_PATH = "./data/baseline_noise.txt"   # File path for baseline low noise levels (leave empty if not available)

# Data analysis start and end date (inclusive), format YYYY-MM-DD
START_DATE = "2020-01-01"
END_DATE   = "2020-04-30"

# Training / testing city split settings
TRAIN_CITY_NAMES = []   # List of city names designated for training (empty to use all cities)
TEST_CITY_NAMES  = []   # List of city names designated for testing (these cities will be excluded from training)
TEST_SIZE        = 0.0  # If TEST_CITY_NAMES is empty, this ratio randomly selects a portion of cities as the test set (0 means no split)
RANDOM_SEED      = 0    # Random seed (for random city splitting)

model_type = "fusion_a"

# Time window length (days) for sequence model input
WINDOW_SIZE = 1

# Hyperparameter search space for CatBoost model
CAT_PARAM_GRID = {
    "iterations": [2000],
    "learning_rate": [0.1, 0.01,  0.001],
    "depth": [3, 5, 7, 9],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0, 1, 2]
}

# Hyperparameter search space for sequence model (LSTM/GRU/Transformer)
SEQ_PARAM_GRID = {
    "hidden_size": [32, 64],
    "num_layers":  [1,  3, 5],
    "batch_size":  [16, 32, 64],
    "learning_rate": [0.1,  0.01],
    "activation": ["linear", "relu"]
}

# Model training switches and settings
USE_CATBOOST       = True      # Whether to train/use CatBoost regression model
USE_SEQUENCE_MODEL = True      # Whether to train/use sequence model (LSTM/GRU/Transformer)
MODEL_TYPE         = "LSTM"    # Sequence model type: "LSTM", "GRU", or "Transformer"
USE_ATTENTION      = False     # If using LSTM/GRU, whether to include Attention mechanism in the model

# Sequence model training parameters
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 5
MULTI_TRAIN_ROUNDS = 50

# Model saving paths
OUTPUT_DIR          = "outputs"
BEST_CAT_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_catboost_model.cbm")
BEST_SEQ_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_sequence_model.pth")
