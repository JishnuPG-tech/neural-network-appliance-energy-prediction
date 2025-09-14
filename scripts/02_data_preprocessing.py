#!/usr/bin/env python3
"""
🔧 Neural Network Data Preprocessing Script
Data preparation pipeline for TensorFlow/Keras neural network training

This script handles all data preprocessing steps including:
- Data cleaning and validation
- Feature engineering
- Scaling and normalization
- Train/test split preparation

Author: Your Name
Date: September 14, 2025
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    """Main preprocessing pipeline."""
    
    print("🔧 Starting Neural Network Data Preprocessing")
    print("=" * 50)
    
    # Step 1: Setup and Imports
    setup_environment()
    
    # Step 2: Load Data
    df = load_data()
    
    # Step 3: Data Cleaning
    df_clean = clean_data(df)
    
    # Step 4: Feature Engineering
    df_engineered = engineer_features(df_clean)
    
    # Step 5: Prepare for Neural Network
    X_train, X_test, y_train, y_test, scaler = prepare_for_neural_network(df_engineered)
    
    # Step 6: Save Processed Data
    save_processed_data(X_train, X_test, y_train, y_test, scaler)
    
    print("\n✅ Data preprocessing completed successfully!")
    return X_train, X_test, y_train, y_test, scaler

def setup_environment():
    """Setup environment and import required libraries."""
    
    print("📚 Setting up preprocessing environment...")
    
    global pd, np, StandardScaler, MinMaxScaler, train_test_split, plt, sns
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("✅ Preprocessing environment ready!")

def load_data():
    """Load data from the exploration phase."""
    
    print("\n📊 Loading data for preprocessing...")
    
    data_path = os.path.join(project_root, 'data', 'appliances_sample_data.csv')
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape}")
    except FileNotFoundError:
        print("⚠️ Creating synthetic data for demonstration...")
        df = create_synthetic_data()
    
    return df

def create_synthetic_data():
    """Create synthetic data if original not available."""
    
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Appliances': np.random.normal(100, 30, n_samples),
        'T1': np.random.normal(20, 5, n_samples),
        'RH_1': np.random.normal(40, 10, n_samples),
        'T2': np.random.normal(22, 4, n_samples),
        'RH_2': np.random.normal(45, 8, n_samples),
        'T_out': np.random.normal(15, 8, n_samples),
        'Press_mm_hg': np.random.normal(760, 20, n_samples),
        'RH_out': np.random.normal(50, 15, n_samples),
        'Windspeed': np.random.normal(5, 2, n_samples),
        'Visibility': np.random.normal(25, 5, n_samples),
        'Tdewpoint': np.random.normal(10, 6, n_samples)
    })
    
    df['Appliances'] = np.abs(df['Appliances'])
    return df

def clean_data(df):
    """Clean and validate the dataset."""
    
    print("\n🧹 Cleaning data...")
    
    df_clean = df.copy()
    
    # Remove any rows with missing values
    initial_shape = df_clean.shape
    df_clean = df_clean.dropna()
    print(f"📊 Removed {initial_shape[0] - df_clean.shape[0]} rows with missing values")
    
    # Remove outliers using IQR method for target variable
    if 'Appliances' in df_clean.columns:
        Q1 = df_clean['Appliances'].quantile(0.25)
        Q3 = df_clean['Appliances'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = df_clean.shape[0]
        df_clean = df_clean[
            (df_clean['Appliances'] >= lower_bound) & 
            (df_clean['Appliances'] <= upper_bound)
        ]
        outliers_removed = outliers_before - df_clean.shape[0]
        print(f"🎯 Removed {outliers_removed} outliers from target variable")
    
    print(f"✅ Clean dataset shape: {df_clean.shape}")
    return df_clean

def engineer_features(df):
    """Create additional features for better neural network performance."""
    
    print("\n⚙️ Engineering features...")
    
    df_eng = df.copy()
    
    # Time-based features if date column exists
    if 'date' in df_eng.columns:
        df_eng['date'] = pd.to_datetime(df_eng['date'])
        df_eng['hour'] = df_eng['date'].dt.hour
        df_eng['day_of_week'] = df_eng['date'].dt.dayofweek
        df_eng['month'] = df_eng['date'].dt.month
        df_eng['is_weekend'] = (df_eng['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng['hour'] / 24)
        df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng['hour'] / 24)
        df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day_of_week'] / 7)
        df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day_of_week'] / 7)
        
        # Drop original date column for neural network
        df_eng = df_eng.drop(['date'], axis=1)
        
        print("🕐 Added time-based features")
    
    # Temperature and humidity interactions
    if all(col in df_eng.columns for col in ['T1', 'T2', 'RH_1', 'RH_2']):
        df_eng['temp_avg'] = (df_eng['T1'] + df_eng['T2']) / 2
        df_eng['humidity_avg'] = (df_eng['RH_1'] + df_eng['RH_2']) / 2
        df_eng['temp_diff'] = df_eng['T1'] - df_eng['T2']
        df_eng['humidity_diff'] = df_eng['RH_1'] - df_eng['RH_2']
        
        print("🌡️ Added temperature and humidity features")
    
    # Comfort index (simplified)
    if all(col in df_eng.columns for col in ['temp_avg', 'humidity_avg']):
        df_eng['comfort_index'] = df_eng['temp_avg'] - (df_eng['humidity_avg'] / 10)
        print("😌 Added comfort index feature")
    
    print(f"✅ Feature engineering completed. New shape: {df_eng.shape}")
    return df_eng

def prepare_for_neural_network(df):
    """Prepare data specifically for neural network training."""
    
    print("\n🧠 Preparing data for neural network...")
    
    # Separate features and target
    target_col = 'Appliances'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"🎯 Features: {len(feature_cols)}")
    print(f"📊 Samples: {len(X)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"🚂 Training set: {X_train.shape}")
    print(f"🧪 Test set: {X_test.shape}")
    
    # Scale the features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to numpy arrays with proper dtypes
    X_train_final = X_train_scaled.astype(np.float32)
    X_test_final = X_test_scaled.astype(np.float32)
    y_train_final = y_train.values.astype(np.float32)
    y_test_final = y_test.values.astype(np.float32)
    
    print("📏 Applied StandardScaler to features")
    print("🔢 Converted to float32 for TensorFlow compatibility")
    
    # Display feature information
    print("\n📋 Neural Network Input Summary:")
    print(f"   Input shape: ({X_train_final.shape[1]},)")
    print(f"   Output shape: (1,)")
    print(f"   Training samples: {len(X_train_final)}")
    print(f"   Test samples: {len(X_test_final)}")
    print(f"   Feature names: {feature_cols}")
    
    return X_train_final, X_test_final, y_train_final, y_test_final, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """Save processed data for model training."""
    
    print("\n💾 Saving processed data...")
    
    # Create data directory if it doesn't exist
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(processed_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_dir, 'y_test.npy'), y_test)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
    
    print(f"✅ Processed data saved to: {processed_dir}")
    
    # Create info file
    info = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open(os.path.join(processed_dir, 'data_info.txt'), 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print("📋 Data info saved")

if __name__ == "__main__":
    main()