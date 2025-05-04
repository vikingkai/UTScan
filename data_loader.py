# data_loader.py
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import logging
import config

def load_data():
    """
    Load and preprocess the data required for urban travel intensity prediction.
    Returns:
        city_names (list): List of city names corresponding to the loaded data
        X_seq_list (list of np.ndarray): Feature sequences for each city (shape: days x features)
        Y_seq_list (list of np.ndarray): Travel intensity sequences for each city (shape: days)
    """
    # 1. Load baseline low-noise levels (if provided)
    baseline_noise = {}
    if os.path.exists(config.LOW_VALUES_PATH):
        with open(config.LOW_VALUES_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    code, vals = parts
                    try:
                        baseline_noise[code] = eval(vals)
                    except Exception as e:
                        logging.warning(f"Could not parse baseline noise values for {code}: {e}")
                        continue

    # 2. Load city–station mapping table
    if not os.path.exists(config.MAPPING_PATH):
        raise FileNotFoundError(f"Mapping table file not found: {config.MAPPING_PATH}")
    df_map = pd.read_excel(config.MAPPING_PATH, dtype=str, engine='openpyxl', header=None)
    mapping = {}
    for _, row in df_map.iterrows():
        station = str(row.iloc[0]).strip().strip("',\"")
        city    = str(row.iloc[1]).strip().strip("',\"")
        if station and city:
            if station in mapping:
                logging.warning(f"Duplicate station code {station} in mapping table; using the last occurrence")
            mapping[station] = city
    if not mapping:
        raise RuntimeError(f"No valid station mapping data found in {config.MAPPING_PATH}")

    # 3. Load travel intensity data table
    if not os.path.exists(config.TRAVEL_DATA_PATH):
        raise FileNotFoundError(f"Travel intensity data file not found: {config.TRAVEL_DATA_PATH}")
    df_travel = pd.read_excel(config.TRAVEL_DATA_PATH, engine='openpyxl', dtype=str)
    # Assume the second column is the city name column and set it as the index
    city_col = df_travel.columns[1]
    df_travel.set_index(city_col, inplace=True)
    df_travel.columns = df_travel.columns.map(str)

    # 4. Generate analysis date range list (list of YYYYMMDD strings)
    start_date = datetime.strptime(config.START_DATE, "%Y-%m-%d")
    end_date   = datetime.strptime(config.END_DATE, "%Y-%m-%d")
    date_range = []
    d = start_date
    while d <= end_date:
        date_range.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    # 5. Search for PSD feature files
    psd_files = glob.glob(os.path.join(config.DATA_DIR, "*_BHZ_*.txt"))
    file_map = {os.path.basename(fp): fp for fp in psd_files}
    logging.info(f"Found {len(psd_files)} PSD files in {config.DATA_DIR}")

    city_names = []
    X_seq_list = []
    Y_seq_list = []

    # 6. Iterate over each station and process data
    target_periods = [0.05, 0.058, 0.068, 0.079, 0.1, 0.11, 0.117, 0.126, 0.155, 0.2, 0.23, 0.317, 0.49]
    for station_code, city_name in mapping.items():
        # Construct the PSD filename for the corresponding year, e.g., "HB_WHN_BHZ_2020.txt"
        net, sta = station_code.split('.', 1)
        year_str = datetime.strftime(start_date, "%Y")
        fname = f"{net}_{sta}_BHZ_{year_str}.txt"
        if fname not in file_map:
            logging.warning(f"PSD file {fname} for station {station_code} not found; skipping")
            continue
        psd_path = file_map[fname]
        logging.info(f"Processing station {station_code} -> city {city_name}")

        # Load the travel intensity series for this city and align the date index
        if city_name not in df_travel.index:
            logging.warning(f"City {city_name} not found in travel intensity data; skipping")
            continue
        travel_series = df_travel.loc[city_name].reindex(date_range)
        travel_series = travel_series.ffill().bfill().astype(float)

        # 6.1 Parse the PSD file hourly, extract target band noise features
        daily_hours = {}
        with open(psd_path, 'r', encoding='utf-8', errors='ignore') as f:
            prev_timestamp = None
            hour_data_pairs = []
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                token = parts[0]
                # Check if this is a new hourly segment marker (format like YYYY_MM_DD_HH)
                if len(token) >= 10 and token[15] == '_' and token[18] == '_':
                    # Aggregate and save features for the previous hour
                    if prev_timestamp is not None:
                        if hour_data_pairs:
                            features = []
                            for T in target_periods:
                                target_p = 1.08 * T
                                closest_noise = None
                                min_diff = float('inf')
                                for (prd_val, noise_val) in hour_data_pairs:
                                    diff = abs(prd_val - target_p)
                                    if diff < min_diff:
                                        min_diff = diff
                                        closest_noise = noise_val
                                features.append(closest_noise if closest_noise is not None else None)
                            day_str = prev_timestamp[:8]  # Extract YYYYMMDD
                            daily_hours.setdefault(day_str, []).append(features)
                        else:
                            logging.debug(f"Station {station_code} ({city_name}) - no valid data for hour {prev_timestamp[:8]}")
                    # Start new hourly segment
                    date_part = token[11:21]  # e.g., YYYY_MM_DD
                    hour_part = token[22:24] if len(token) >= 13 else token[21:23]
                    day_str = date_part.replace('_', '')
                    current_timestamp = day_str + (hour_part if hour_part else '')
                    prev_timestamp = current_timestamp
                    hour_data_pairs = []
                    data_str = raw_line.split(None, 1)[1] if len(raw_line.split(None, 1)) > 1 else ""
                    data_str = data_str.strip()
                else:
                    # Continuing lines for the current hour segment
                    if prev_timestamp is None:
                        continue  # Skip lines without timestamp (incomplete data)
                    data_str = line
                # Parse all "period;noise" pairs in this line
                if data_str:
                    segments = data_str.split('&')
                    for seg in segments:
                        seg = seg.strip()
                        if not seg:
                            continue
                        if ';' not in seg:
                            logging.debug(f"Malformed PSD segment: {seg} ({station_code} at hour {prev_timestamp[8:]})")
                            continue
                        prd_str, noise_str = seg.split(';', 1)
                        try:
                            prd_val = float(prd_str)
                        except Exception as e:
                            logging.debug(f"Cannot parse period '{prd_str}' ({station_code} at hour {prev_timestamp[8:]}): {e}")
                            continue
                        if prd_val <= 0:
                            continue
                        if noise_str.strip().lower() == '-inf':
                            # Infinite noise value indicates no data; skip
                            continue
                        try:
                            noise_val = float(noise_str)
                        except Exception as e:
                            logging.debug(f"Cannot parse noise '{noise_str}' ({station_code} at hour {prev_timestamp[8:]}): {e}")
                            continue
                        # Only consider period data within 0.05s–0.55s
                        if prd_val < 0.05 or prd_val > 0.55:
                            continue
                        hour_data_pairs.append((prd_val, noise_val))
            # Process the last hour segment at end of file
            if prev_timestamp is not None:
                if hour_data_pairs:
                    features = []
                    for T in target_periods:
                        target_p = 1.08 * T
                        closest_noise = None
                        min_diff = float('inf')
                        for (prd_val, noise_val) in hour_data_pairs:
                            diff = abs(prd_val - target_p)
                            if diff < min_diff:
                                min_diff = diff
                                closest_noise = noise_val
                        features.append(closest_noise if closest_noise is not None else None)
                    day_str = prev_timestamp[:8]
                    daily_hours.setdefault(day_str, []).append(features)
                    logging.debug(f"Station {station_code} ({city_name}) - last hour features for {day_str}: {features}")
                else:
                    logging.debug(f"Station {station_code} ({city_name}) - no valid data for hour {prev_timestamp[:8]}")

        # 6.2 Aggregate hourly features into daily features (mean)
        daily_features = {}
        for day, features_list in daily_hours.items():
            if not features_list:
                continue
            df_day = pd.DataFrame(features_list)
            daily_feat = df_day.mean(axis=0).tolist()  # Compute mean by column
            daily_features[day] = daily_feat
        if not daily_features:
            logging.warning(f"No PSD feature data for station {station_code}; skipping")
            continue

        # 6.3 Build feature DataFrame for this city and align to the full date sequence
        df_features = pd.DataFrame.from_dict(daily_features, orient='index')
        df_features = df_features.reindex(date_range).ffill().bfill()

        # 6.4 If a baseline noise curve is provided, perform baseline subtraction normalization
        if station_code in baseline_noise and len(baseline_noise[station_code]) == df_features.shape[1]:
            baseline_vals = baseline_noise[station_code]
            df_features = df_features - baseline_vals

        # 6.5 Save the features and target series for this city
        city_names.append(city_name)
        X_seq_list.append(df_features.values.astype(float))
        Y_seq_list.append(travel_series.values.astype(float))

    logging.info(f"Loaded data for {len(city_names)} cities in total")
    return city_names, X_seq_list, Y_seq_list
