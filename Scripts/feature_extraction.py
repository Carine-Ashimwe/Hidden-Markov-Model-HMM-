import os
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.metrics import classification_report

# ==========================================
# 1. CORE FEATURE EXTRACTION
# ==========================================
def extract_features(window_df, fs):
    """Extracts 6 features: Variance, SMA, and Dominant Frequency."""
    ax, ay, az = window_df['accel_x'].values, window_df['accel_y'].values, window_df['accel_z'].values
    gx, gy, gz = window_df['gyro_x'].values, window_df['gyro_y'].values, window_df['gyro_z'].values
    
    # Magnitudes (combines x, y, z into one overall intensity vector)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # --- TIME-DOMAIN ---
    accel_var = np.var(accel_mag)
    gyro_var = np.var(gyro_mag)
    
    accel_sma = np.sum(np.abs(ax) + np.abs(ay) + np.abs(az)) / len(window_df)
    gyro_sma = np.sum(np.abs(gx) + np.abs(gy) + np.abs(gz)) / len(window_df)
    
    # --- FREQUENCY-DOMAIN (FFT) ---
    def get_dominant_frequency(signal_mag):
        N = len(signal_mag)
        fft_values = np.abs(rfft(signal_mag))
        frequencies = rfftfreq(N, d=1/fs)
        if len(fft_values) > 1:
            dom_freq_idx = np.argmax(fft_values[1:]) + 1 
            return frequencies[dom_freq_idx]
        return 0.0

    accel_dom_freq = get_dominant_frequency(accel_mag)
    gyro_dom_freq = get_dominant_frequency(gyro_mag)
        
    return [accel_var, gyro_var, accel_sma, gyro_sma, accel_dom_freq, gyro_dom_freq]

# ==========================================
# 2. SLIDING WINDOW & SENSOR FUSION
# ==========================================
def process_user_data(user_folder, fs, activity_map, window_size_sec=2.0, overlap_sec=1.0):
    all_features = []
    all_labels = []
    lengths = [] 

    window_size_rows = int(window_size_sec * fs)
    step_size_rows = int((window_size_sec - overlap_sec) * fs)
    
    # Loop through the Activity folders (e.g., jumping, Standing, etc.)
    for folder_name in os.listdir(user_folder):
        activity_path = os.path.join(user_folder, folder_name)
        if not os.path.isdir(activity_path): continue
        
        # Match folder name case-insensitively (handles "jumping" vs "Jumping")
        label = next((v for k, v in activity_map.items() if k.lower() == folder_name.lower()), None)
        if label is None:
            continue # Skip folders that aren't in our activity map
            
        # Loop through Session folders (e.g., jumping_1, jumping_2)
        for session in os.listdir(activity_path):
            session_path = os.path.join(activity_path, session)
            if not os.path.isdir(session_path): continue
            
            acc_file = os.path.join(session_path, 'Accelerometer.csv')
            gyro_file = os.path.join(session_path, 'Gyroscope.csv')
            
            if os.path.exists(acc_file) and os.path.exists(gyro_file):
                # Read CSVs and rename columns immediately
                df_a = pd.read_csv(acc_file).rename(columns={'x':'accel_x','y':'accel_y','z':'accel_z'})
                df_g = pd.read_csv(gyro_file).rename(columns={'x':'gyro_x','y':'gyro_y','z':'gyro_z'})
                
                # Merge based on seconds_elapsed (Sensor Fusion)
                merged = pd.merge_asof(
                    df_a.sort_values('seconds_elapsed'), 
                    df_g[['seconds_elapsed', 'gyro_x', 'gyro_y', 'gyro_z']].sort_values('seconds_elapsed'), 
                    on='seconds_elapsed', 
                    direction='nearest'
                )
                
                # Slide the window
                file_features = []
                for start in range(0, len(merged) - window_size_rows + 1, step_size_rows):
                    window = merged.iloc[start : start + window_size_rows]
                    file_features.append(extract_features(window, fs))
                
                if file_features:
                    all_features.extend(file_features)
                    all_labels.extend([label] * len(file_features))
                    lengths.append(len(file_features))
                    
    return np.array(all_features), np.array(all_labels), lengths

# ==========================================
# 3. DATA LOADING & MERGING
# ==========================================
ACTIVITY_MAP = {'Standing': 0, 'Walking': 1, 'Jumping': 2, 'Still': 3}

# Assuming you run this from the root workspace folder (Hidden-Markov-Model-HMM-)
base_path_armstrong = 'data/Armstrong/' 
base_path_carine = 'data/Carine/'

# Process the data (assuming 50Hz for both, adjust if Carine's is different)
print("Processing Armstrong's data...")
X_arm, y_arm, len_arm = process_user_data(base_path_armstrong, fs=50, activity_map=ACTIVITY_MAP)

print("Processing Carine's data...")
X_car, y_car, len_car = process_user_data(base_path_carine, fs=50, activity_map=ACTIVITY_MAP)

# Combine both datasets vertically
X_raw = np.vstack((X_arm, X_car))
y_train = np.concatenate((y_arm, y_car))
train_lengths = len_arm + len_car

# ==========================================
# 4. SAVE EXTRACTED FEATURES TO CSV
# ==========================================
print("\nSaving merged feature dataset to CSV...")
feature_columns = [
    'accel_variance', 'gyro_variance', 
    'accel_sma', 'gyro_sma', 
    'accel_dom_freq', 'gyro_dom_freq'
]

# Create a DataFrame
df_features = pd.DataFrame(X_raw, columns=feature_columns)
df_features['activity_code'] = y_train

# Map the numbers back to text names for a well-labeled dataset
reverse_map = {v: k for k, v in ACTIVITY_MAP.items()}
df_features['activity_name'] = df_features['activity_code'].map(reverse_map)

# Save it to the root data folder
output_path = 'data/merged_features_dataset.csv'
df_features.to_csv(output_path, index=False)
print(f"✅ Merged dataset successfully saved to: {output_path}")
