"""
Enhanced data preprocessing for rakat detection.
Supports 6-axis sensor data and multi-class rakat classification (0=non-namaz, 1=2_rakat, 2=3_rakat, 3=4_rakat).

Extracts 100+ features including rakat-specific cycle detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.stats import entropy

def extract_rakat_features(df):
    """
    Extract rakat-specific features for cycle detection.
    These features help identify patterns of repetitive prayer movements.
    """
    features = {}
    
    # Get gyroscope magnitude for cycle detection
    gyro_magnitude = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    acc_magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # Detect movement peaks (possible ruku/sujud)
    from scipy.signal import find_peaks
    gyro_peaks, _ = find_peaks(gyro_magnitude, height=np.mean(gyro_magnitude) + np.std(gyro_magnitude), distance=20)
    acc_peaks, _ = find_peaks(acc_magnitude, height=np.mean(acc_magnitude) + np.std(acc_magnitude), distance=20)
    
    # Rakat counting features
    features['gyro_peak_count'] = len(gyro_peaks)
    features['acc_peak_count'] = len(acc_peaks)
    features['total_peaks'] = len(gyro_peaks) + len(acc_peaks)
    
    # Duration and timing features
    duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    features['duration_seconds'] = duration
    features['peaks_per_minute'] = (len(gyro_peaks) + len(acc_peaks)) / (duration / 60) if duration > 0 else 0
    
    # Pattern regularity (how regular are the cycles)
    if len(gyro_peaks) > 2:
        peak_intervals = np.diff(gyro_peaks / 50)  # Convert sample indices to seconds
        features['peak_interval_mean'] = np.mean(peak_intervals)
        features['peak_interval_std'] = np.std(peak_intervals)
        features['peak_regularity'] = 1 / (1 + np.std(peak_intervals))  # Higher for regular intervals
    else:
        features['peak_interval_mean'] = 0
        features['peak_interval_std'] = 0
        features['peak_regularity'] = 0
    
    # Movement segmentation features
    # Detect major movement changes (transitions between prayer positions)
    gyro_diff = np.abs(np.diff(gyro_magnitude))
    movement_segments = gyro_diff > np.mean(gyro_diff) + np.std(gyro_diff)
    features['movement_segments'] = np.sum(movement_segments)
    features['movement_frequency'] = np.sum(movement_segments) / len(gyro_diff) if len(gyro_diff) > 0 else 0
    
    # Energy distribution (helps identify prayer structure)
    window_size = min(100, len(gyro_magnitude) // 4)  # Adaptive window size
    if len(gyro_magnitude) >= window_size:
        # Divide signal into segments
        segments = [gyro_magnitude[i:i+window_size] for i in range(0, len(gyro_magnitude) - window_size, window_size)]
        segment_energies = [np.sum(seg**2) for seg in segments]
        
        features['segment_energy_mean'] = np.mean(segment_energies)
        features['segment_energy_std'] = np.std(segment_energies)
        features['segment_energy_max'] = np.max(segment_energies)
        features['energy_variation'] = np.std(segment_energies) / (np.mean(segment_energies) + 1e-8)
    else:
        features['segment_energy_mean'] = np.sum(gyro_magnitude**2)
        features['segment_energy_std'] = 0
        features['segment_energy_max'] = np.sum(gyro_magnitude**2)
        features['energy_variation'] = 0
    
    # Signal entropy (higher for complex patterns like longer prayers)
    hist, _ = np.histogram(gyro_magnitude, bins=20)
    features['signal_entropy'] = entropy(hist + 1e-8)  # Add small value to avoid log(0)
    
    # Combined rakat score (heuristic for rakat prediction)
    # More peaks + longer duration + regular intervals = likely more rakats
    base_score = (len(gyro_peaks) + len(acc_peaks)) * 0.1
    duration_score = min(duration / 120, 3) * 0.3  # Cap at 3 for 4+ minutes
    regularity_score = features['peak_regularity'] * 0.2
    complexity_score = features['signal_entropy'] * 0.1
    
    features['rakat_score'] = base_score + duration_score + regularity_score + complexity_score
    
    return features

def extract_features(df):
    """
    Extract statistical features from 6-axis sensor data (gyroscope + accelerometer).
    
    Enhanced feature extraction combining:
    - Gyroscope: Rotation rate (angular velocity)
    - Accelerometer: Linear acceleration + gravity (orientation + movement)
    - Cross-features: Combined sensor patterns
    - Rakat features: Cycle detection and pattern recognition
    """
    features = {}
    
    # Check if we have accelerometer data
    has_accelerometer = all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z'])
    
    # Extract features for gyroscope axes
    for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
        data = df[axis].values
        
        # Statistical features
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_median'] = np.median(data)
        features[f'{axis}_q25'] = np.percentile(data, 25)
        features[f'{axis}_q75'] = np.percentile(data, 75)
        
        # Energy (sum of squares)
        features[f'{axis}_energy'] = np.sum(data ** 2)
        
        # Absolute mean (ignoring direction)
        features[f'{axis}_abs_mean'] = np.mean(np.abs(data))
        
        # Zero crossing rate (how often signal changes direction)
        features[f'{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
    
    # Extract features for accelerometer axes (if available)
    if has_accelerometer:
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            data = df[axis].values
            
            # Statistical features
            features[f'{axis}_mean'] = np.mean(data)
            features[f'{axis}_std'] = np.std(data)
            features[f'{axis}_min'] = np.min(data)
            features[f'{axis}_max'] = np.max(data)
            features[f'{axis}_median'] = np.median(data)
            features[f'{axis}_q25'] = np.percentile(data, 25)
            features[f'{axis}_q75'] = np.percentile(data, 75)
            
            # Energy (sum of squares)
            features[f'{axis}_energy'] = np.sum(data ** 2)
            
            # Absolute mean (ignoring direction)
            features[f'{axis}_abs_mean'] = np.mean(np.abs(data))
            
            # Zero crossing rate
            features[f'{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
            
            # Gravity component (low-frequency part)
            fs = 50  # Assume 50Hz sampling rate
            cutoff = 0.3  # Low-pass cutoff for gravity
            b, a = signal.butter(4, cutoff/(fs/2), 'low')
            gravity_component = signal.filtfilt(b, a, data)
            linear_acceleration = data - gravity_component
            
            features[f'{axis}_gravity_mean'] = np.mean(gravity_component)
            features[f'{axis}_gravity_std'] = np.std(gravity_component)
            features[f'{axis}_linear_mean'] = np.mean(linear_acceleration)
            features[f'{axis}_linear_std'] = np.std(linear_acceleration)
    
    # Cross-axis features for gyroscope
    features['gyro_magnitude_mean'] = np.mean(np.sqrt(
        df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2
    ))
    
    features['gyro_magnitude_std'] = np.std(np.sqrt(
        df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2
    ))
    
    # Cross-axis features for accelerometer (if available)
    if has_accelerometer:
        features['acc_magnitude_mean'] = np.mean(np.sqrt(
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        ))
        
        features['acc_magnitude_std'] = np.std(np.sqrt(
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        ))
        
        # Combined 6-axis magnitude
        total_magnitude = np.sqrt(
            df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2 +
            df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
        )
        features['total_magnitude_mean'] = np.mean(total_magnitude)
        features['total_magnitude_std'] = np.std(total_magnitude)
        
        # Cross-correlation features between gyro and accel
        for gyro_axis in ['gyro_x', 'gyro_y', 'gyro_z']:
            for acc_axis in ['acc_x', 'acc_y', 'acc_z']:
                correlation = np.corrcoef(df[gyro_axis].values, df[acc_axis].values)[0, 1]
                if not np.isnan(correlation):
                    features[f'{gyro_axis}_{acc_axis}_corr'] = correlation
                else:
                    features[f'{gyro_axis}_{acc_axis}_corr'] = 0
    
    # Add rakat-specific features
    rakat_features = extract_rakat_features(df)
    features.update(rakat_features)
    
    return features

def load_and_preprocess_data(data_dir='data'):
    """
    Load all CSV files and extract features with rakat labels.
    
    Returns:
        X: Feature matrix (samples x features)
        y: Labels (0=non_namaz, 1=2_rakat, 2=3_rakat, 3=4_rakat)
        feature_names: Names of the features
        subcategories: List of subcategory names for each sample
    """
    data_path = Path(data_dir)
    
    all_features = []
    all_labels = []
    subcategories = []
    
    print("Loading and preprocessing data...")
    print("-" * 50)
    
    # Define subfolder structure
    non_namaz_subfolders = ['walking', 'running', 'sitting_floor', 'sitting_chair']
    rakat_subfolders = ['2_rakat', '3_rakat', '4_rakat']
    namaz_subcategories = ['standing_loose_clothing', 'standing_tight_clothing', 'sitting_floor']
    
    total_files = 0
    
    # Load non-namaz data (label = 0)
    for subfolder in non_namaz_subfolders:
        subfolder_path = data_path / 'non_namaz' / subfolder
        if subfolder_path.exists():
            files = list(subfolder_path.glob('*.csv'))
            print(f"Loading {len(files)} {subfolder} non-namaz samples...")
            total_files += len(files)
            
            for i, file_path in enumerate(files):
                df = pd.read_csv(file_path)
                features = extract_features(df)
                all_features.append(features)
                all_labels.append(0)  # 0 = non-namaz
                subcategories.append(f"non_namaz_{subfolder}")
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(files)} {subfolder} files...")
    
    # Load namaz data for each rakat count (labels 1, 2, 3)
    for rakat_count, rakat_folder in enumerate(rakat_subfolders, start=1):
        for subcategory in namaz_subcategories:
            subfolder_path = data_path / 'namaz' / rakat_folder / subcategory
            if subfolder_path.exists():
                files = list(subfolder_path.glob('*.csv'))
                print(f"Loading {len(files)} {rakat_folder} {subcategory} samples...")
                total_files += len(files)
                
                for i, file_path in enumerate(files):
                    df = pd.read_csv(file_path)
                    features = extract_features(df)
                    all_features.append(features)
                    all_labels.append(rakat_count)  # 1=2_rakat, 2=3_rakat, 3=4_rakat
                    subcategories.append(f"namaz_{rakat_folder}_{subcategory}")
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i+1}/{len(files)} {subcategory} files...")
    
    # Convert to DataFrame then to numpy arrays
    features_df = pd.DataFrame(all_features)
    X = features_df.values
    y = np.array(all_labels)
    feature_names = features_df.columns.tolist()
    
    print("-" * 50)
    print(f"✓ Preprocessing complete!")
    print(f"  - Total samples: {len(y)}")
    print(f"  - Non-namaz samples: {np.sum(y == 0)}")
    print(f"  - 2 rakat samples: {np.sum(y == 1)}")
    print(f"  - 3 rakat samples: {np.sum(y == 2)}")
    print(f"  - 4 rakat samples: {np.sum(y == 3)}")
    
    # Show distribution by category
    print(f"  - Namaz samples breakdown:")
    for subcategory in subcategories:
        if 'namaz_' in subcategory:
            print(f"    • {subcategory}: 1 sample")
    
    print(f"  - Non-namaz samples breakdown:")
    non_namaz_count = 0
    for subcategory in subcategories:
        if 'non_namaz_' in subcategory:
            non_namaz_count += 1
    print(f"    • Total: {non_namaz_count} samples")
    
    print(f"  - Features extracted: {len(feature_names)}")
    
    # Check if we have accelerometer features
    has_accel_features = any('acc_' in feature for feature in feature_names)
    if has_accel_features:
        acc_features = [f for f in feature_names if 'acc_' in f and 'corr' not in f]
        gyro_features = [f for f in feature_names if 'gyro_' in f and 'corr' not in f]
        cross_features = [f for f in feature_names if any(keyword in f for keyword in ['magnitude', 'total', '_corr'])]
        print(f"    • Gyroscope features: {len(gyro_features)}")
        print(f"    • Accelerometer features: {len(acc_features)}")
        print(f"    • Combined features: {len(cross_features)}")
        print(f"    • Enhanced 6-axis data detected ✓")
    else:
        print(f"    • Gyroscope-only data (3-axis)")
    
    return X, y, feature_names, subcategories

def preprocess_single_file(file_path):
    """
    Preprocess a single CSV file for prediction.
    
    Args:
        file_path: Path to the data file (.csv)
    
    Returns:
        Feature vector as numpy array
    """
    df = pd.read_csv(file_path)
    features = extract_features(df)
    
    # Convert to DataFrame to ensure correct feature order
    features_df = pd.DataFrame([features])
    return features_df.values[0]