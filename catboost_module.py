# catboost_module.py
# 提供CatBoost模型的训练方法（网格搜索超参数）

import logging
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger("catboost_module")

def train_catboost(X_train, y_train, X_val, y_val, param_grid, cat_features_idx=None):
    """
    网格搜索CatBoostRegressor超参数，在验证集上评估，返回最佳模型及参数。
    X_train, X_val: numpy数组或CatBoost支持的数据结构
    cat_features_idx: 类别型特征的列索引列表（若有分类特征）
    """
    from itertools import product
    keys = list(param_grid.keys())
    results = []
    combos = [dict(zip(keys, values)) for values in product(*(param_grid[k] for k in keys))]
    best_model = None
    best_params = None
    best_rmse = float('inf')
    for idx, params in enumerate(combos, start=1):
        logger.info(f"Training CatBoost combo {idx}/{len(combos)}: {params}")
        params_full = params.copy()
        # 设置缺省参数
        params_full.setdefault('loss_function', 'RMSE')
        params_full.setdefault('random_seed', 0)
        params_full.setdefault('verbose', False)
        # 初始化模型
        model = CatBoostRegressor(**params_full)
        # 如果有类别特征，需要使用Pool传入
        if cat_features_idx is not None:
            from catboost import Pool
            train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
            val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True, verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True, verbose=False)
        # 验证集预测
        pred_val = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        mae = mean_absolute_error(y_val, pred_val)
        r2 = r2_score(y_val, pred_val)
        # 计算其他指标
        if len(y_val) > 0:
            mape = np.mean(np.abs((y_val - pred_val) / y_val)) * 100
            if len(y_val) > 1:
                corr = np.corrcoef(y_val, pred_val)[0, 1]
            else:
                corr = 0.0
        else:
            mape = 0.0
            corr = 0.0
        logger.info(f"Params {params_full} -> Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%, r={corr:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params_full
        results.append((params, rmse, mae, r2, mape, corr))
    logger.info(f"Best CatBoost params: {best_params}, Validation RMSE={best_rmse:.4f}")
    return best_model, best_params, results
