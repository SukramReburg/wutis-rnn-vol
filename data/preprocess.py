from indicators import *
import yaml
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from numpy.lib.stride_tricks import sliding_window_view

def merge_data(): 
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    tickers = config['indicators_for_tickers']

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = config['paths']
    path = os.path.join(base_dir, paths['raw'])
    
    # Read and merge all CSV files
    merged_data = pd.DataFrame()
    for ticker in tickers:
        file_path = os.path.join(path, f"{ticker}.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data.columns = [f"{ticker}_{col}" if col != 'timestamp' else 'timestamp' for col in data.columns]
            if merged_data.empty:
                merged_data = data
            else:
                merged_data = pd.merge(merged_data, data, on='timestamp', how='outer')
        else:
            print(f"Processed file for {ticker} not found.")
    
    # Ensure there are no missing values
    merged_data.ffill(inplace=True)
    merged_data.dropna(inplace=True)

    # Drop string columns that are not needed
    string_columns = merged_data.select_dtypes(include=['object']).columns
    merged_data.drop(columns=string_columns, inplace=True) 
   
    # Save the merged data
    print(f"Merged data shape: {merged_data.shape}")
    # merged_data.to_csv(os.path.join(path, 'merged_data.csv'), index=False)

    return merged_data

def create_datasets(merged_data, path):
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    column_to_predict = config['y_column']
    sequence_length = config['sequence_length']

    if column_to_predict not in merged_data.columns:
        raise ValueError(f"Column to predict '{column_to_predict}' not found in the data.")

    # Find the index of the column_to_predict and write it to the YAML file
    column_index = merged_data.columns.get_loc(column_to_predict)
    config['y_column_index'] = column_index
    with open('config/data_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    print(f"Column to predict '{column_to_predict}' found at index {column_index}. Updated config file.")
    print(f"Creating datasets with sequence length: {sequence_length} and target column: {column_to_predict}")

    # Prepare the dataset
    data_values = merged_data.drop(columns=[column_to_predict]).values
    target_values = merged_data[column_to_predict].values

    X = sliding_window_view(data_values, (sequence_length, data_values.shape[1]), axis=(0, 1))
    X = X.reshape(-1, sequence_length, data_values.shape[1])
    X = X[:-1]  # drop last window without a corresponding target

    y = target_values[sequence_length:]
    
    X = np.array(X)
    y = np.array(y)

    print(f"Dataset created with shape X: {X.shape}, y: {y.shape}")

    # Split the dataset into training, validation, and test sets
    train_ratio = config['train_size']
    val_ratio = config['val_size']
    test_ratio = 1 - train_ratio - val_ratio

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Train, validation, and test ratios must be between 0 and 1 and sum to 1.")

    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    print(f"Dataset split: {train_end} train, {val_end - train_end} val, {total_samples - val_end} test")
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    print(f"Shapes after split - X_train: {X_train.shape}, y_train: {y_train.shape}, "
          f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
          f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Scale the data
    n_features = X_train.shape[2]
    X_train_scaled = np.empty_like(X_train)
    X_val_scaled = np.empty_like(X_val)
    X_test_scaled = np.empty_like(X_test)

    print(f"Scaling data with {n_features} features...")
    feature_scalers = []
    for i in range(n_features):
        scaler = MinMaxScaler()
        X_train_scaled[:, :, i] = scaler.fit_transform(X_train[:, :, i])
        X_val_scaled[:, :, i] = scaler.transform(X_val[:, :, i])
        X_test_scaled[:, :, i] = scaler.transform(X_test[:, :, i])
        feature_scalers.append(scaler)
    
    X_train_scaled = X_train_scaled[..., np.newaxis]
    X_val_scaled = X_val_scaled[..., np.newaxis]
    X_test_scaled = X_test_scaled[..., np.newaxis]
    
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    save_scalers(feature_scalers, y_scaler, path)

    # Save the splits
    np.save(os.path.join(path, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(path, 'y_train.npy'), y_train_scaled)
    np.save(os.path.join(path, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(path, 'y_val.npy'), y_val_scaled)
    np.save(os.path.join(path, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(path, 'y_test.npy'), y_test_scaled)

def save_scalers(feature_scalers, y_scaler, path):
    scaler_dir = os.path.join(path, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(feature_scalers, os.path.join(scaler_dir, 'feature_scalers.joblib'))
    joblib.dump(y_scaler, os.path.join(scaler_dir, 'y_scaler.joblib'))

def load_scalers(path):
    scaler_dir = os.path.join(path, 'scalers')
    feature_scalers = joblib.load(os.path.join(scaler_dir, 'feature_scalers.joblib'))
    y_scaler = joblib.load(os.path.join(scaler_dir, 'y_scaler.joblib'))
    return feature_scalers, y_scaler

def scale_with_saved_scalers(X, feature_scalers):
    X_scaled = np.empty_like(X)
    for i, scaler in enumerate(feature_scalers):
        X_scaled[:, :, i] = scaler.transform(X[:, :, i])
    return X_scaled

def create_dir(): 
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['processed'])
    os.makedirs(path, exist_ok=True)
    return path

def preprocess(): 
    merged_data = merge_data()
    path = create_dir()
    create_datasets(merged_data, path)
    print("Data preprocessing completed successfully.")

    
if __name__ == "__main__":
    preprocess()
