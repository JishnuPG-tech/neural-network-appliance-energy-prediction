#!/usr/bin/env python3
"""
🔍 Neural Network Appliance Energy Prediction - Data Exploration Script
Advanced Deep Learning Project for Individual Appliance Energy Consumption Prediction

This script contains all the data exploration code from the Jupyter notebook
converted to a standalone Python script for PyCharm execution.

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
    """Main function to run all data exploration steps."""
    
    print("🔍 Starting Neural Network Data Exploration")
    print("=" * 50)
    
    # Step 1: Setup and Imports
    setup_environment()
    
    # Step 2: Load and Examine Data
    df = load_data()
    
    # Step 3: Basic Data Analysis
    basic_analysis(df)
    
    # Step 4: Visualizations
    create_visualizations(df)
    
    # Step 5: Correlation Analysis
    correlation_analysis(df)
    
    # Step 6: Statistical Analysis
    statistical_analysis(df)
    
    print("\n✅ Data exploration completed successfully!")

def setup_environment():
    """Setup environment and import required libraries."""
    
    print("📚 Setting up environment and importing libraries...")
    
    global pd, np, plt, sns, px, go, make_subplots
    
    # Data Science Libraries
    import pandas as pd
    import numpy as np
    
    # Visualization Libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Statistical Libraries
    from scipy import stats
    
    # Configure plotting
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    sns.set_palette("husl")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    print("✅ Environment setup completed!")

def load_data():
    """Load and prepare data for analysis."""
    
    print("\n📊 Loading and preparing data...")
    
    # Load the data
    data_path = os.path.join(project_root, 'data', 'appliances_sample_data.csv')
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully from {data_path}")
        print(f"📊 Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("⚠️ Sample data not found. Creating synthetic dataset for demonstration...")
        df = create_synthetic_data()
        
    return df

def create_synthetic_data():
    """Create synthetic appliance energy data for demonstration."""
    
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Appliances': np.random.normal(100, 30, n_samples),  # Target variable
        'T1': np.random.normal(20, 5, n_samples),            # Temperature 1
        'RH_1': np.random.normal(40, 10, n_samples),         # Humidity 1
        'T2': np.random.normal(22, 4, n_samples),            # Temperature 2
        'RH_2': np.random.normal(45, 8, n_samples),          # Humidity 2
        'T_out': np.random.normal(15, 8, n_samples),         # Outside Temperature
        'Press_mm_hg': np.random.normal(760, 20, n_samples), # Pressure
        'RH_out': np.random.normal(50, 15, n_samples),       # Outside Humidity
        'Windspeed': np.random.normal(5, 2, n_samples),      # Wind Speed
        'Visibility': np.random.normal(25, 5, n_samples),    # Visibility
        'Tdewpoint': np.random.normal(10, 6, n_samples)      # Dew Point
    })
    
    # Ensure positive values for Appliances
    df['Appliances'] = np.abs(df['Appliances'])
    
    print("🔧 Synthetic dataset created for demonstration!")
    return df

def basic_analysis(df):
    """Perform basic data analysis."""
    
    print("\n📋 Basic Data Analysis")
    print("-" * 30)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nStatistical summary:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

def create_visualizations(df):
    """Create comprehensive visualizations."""
    
    print("\n📊 Creating visualizations...")
    
    # Distribution plots
    plt.figure(figsize=(15, 10))
    
    # Target variable distribution
    plt.subplot(2, 3, 1)
    plt.hist(df['Appliances'], bins=50, alpha=0.7, color='skyblue')
    plt.title('Distribution of Appliances Energy Consumption')
    plt.xlabel('Energy (Wh)')
    plt.ylabel('Frequency')
    
    # Temperature distributions
    plt.subplot(2, 3, 2)
    plt.hist(df['T1'], bins=30, alpha=0.7, color='lightgreen', label='T1')
    plt.hist(df['T2'], bins=30, alpha=0.7, color='orange', label='T2')
    plt.title('Temperature Distributions')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Humidity distributions
    plt.subplot(2, 3, 3)
    plt.hist(df['RH_1'], bins=30, alpha=0.7, color='pink', label='RH_1')
    plt.hist(df['RH_2'], bins=30, alpha=0.7, color='purple', label='RH_2')
    plt.title('Humidity Distributions')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Time series plot
    plt.subplot(2, 3, 4)
    if 'date' in df.columns:
        plt.plot(df['date'][:100], df['Appliances'][:100])
        plt.title('Appliances Energy Over Time (First 100 hours)')
        plt.xlabel('Time')
        plt.ylabel('Energy (Wh)')
        plt.xticks(rotation=45)
    
    # Box plot for outliers
    plt.subplot(2, 3, 5)
    df.boxplot(column='Appliances', ax=plt.gca())
    plt.title('Appliances Energy Box Plot')
    plt.ylabel('Energy (Wh)')
    
    # Scatter plot example
    plt.subplot(2, 3, 6)
    plt.scatter(df['T1'], df['Appliances'], alpha=0.5)
    plt.title('Appliances vs Temperature T1')
    plt.xlabel('Temperature T1 (°C)')
    plt.ylabel('Energy (Wh)')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizations created!")

def correlation_analysis(df):
    """Perform correlation analysis."""
    
    print("\n🔗 Correlation Analysis")
    print("-" * 25)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print correlations with target variable
    target_corr = corr_matrix['Appliances'].sort_values(ascending=False)
    print("\nCorrelations with Appliances (target variable):")
    for feature, corr_val in target_corr.items():
        if feature != 'Appliances':
            print(f"{feature:15}: {corr_val:6.3f}")

def statistical_analysis(df):
    """Perform statistical analysis."""
    
    print("\n📊 Statistical Analysis")
    print("-" * 23)
    
    # Basic statistics for target variable
    target = df['Appliances']
    
    print(f"Target Variable (Appliances) Statistics:")
    print(f"Mean:           {target.mean():.2f}")
    print(f"Median:         {target.median():.2f}")
    print(f"Standard Dev:   {target.std():.2f}")
    print(f"Min:            {target.min():.2f}")
    print(f"Max:            {target.max():.2f}")
    print(f"Skewness:       {target.skew():.2f}")
    print(f"Kurtosis:       {target.kurtosis():.2f}")
    
    # Feature importance for neural network
    print("\n🧠 Neural Network Preparation Insights:")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col != 'Appliances']
    
    print(f"✅ Features for neural network: {len(features)}")
    print(f"🎯 Target variable: Appliances")
    print(f"📊 Dataset size: {len(df)} samples")
    
    if len(features) > 0:
        print(f"🔧 Recommended preprocessing: StandardScaler or MinMaxScaler")
        print(f"🧠 Neural network architecture suggestion:")
        print(f"   - Input layer: {len(features)} neurons")
        print(f"   - Hidden layers: 2-3 layers with 64-128 neurons each")
        print(f"   - Output layer: 1 neuron (regression)")
        print(f"   - Activation: ReLU for hidden, Linear for output")
        print(f"   - Loss: Mean Squared Error (MSE)")

if __name__ == "__main__":
    main()