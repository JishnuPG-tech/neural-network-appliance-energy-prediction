"""
Utility Functions for Appliance Energy Prediction Project

This module contains helper functions for data handling, visualization,
appliance-specific calculations, and general utilities used throughout the project.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")


def save_data(data: Any, file_path: str, file_format: str = 'csv') -> None:
    """
    Save data to file in specified format.
    
    Args:
        data: Data to save
        file_path (str): Path to save the file
        file_format (str): Format to save ('csv', 'json', 'pickle')
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(file_path)
    create_directory(directory)
    
    try:
        if file_format.lower() == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, pd.Series):
                data.to_csv(file_path, index=False, header=True)
            else:
                pd.DataFrame(data).to_csv(file_path, index=False)
                
        elif file_format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        elif file_format.lower() == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        logger.info(f"Data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise


def load_data(file_path: str, file_format: str = 'csv') -> Any:
    """
    Load data from file.
    
    Args:
        file_path (str): Path to the file
        file_format (str): Format of the file ('csv', 'json', 'pickle')
        
    Returns:
        Any: Loaded data
    """
    try:
        if file_format.lower() == 'csv':
            return pd.read_csv(file_path)
            
        elif file_format.lower() == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
                
        elif file_format.lower() == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def save_model(model: Any, file_path: str) -> None:
    """
    Save a trained model to file.
    
    Args:
        model: Trained model object
        file_path (str): Path to save the model
    """
    save_data(model, file_path, 'pickle')


def load_model(file_path: str) -> Any:
    """
    Load a trained model from file.
    
    Args:
        file_path (str): Path to the model file
        
    Returns:
        Any: Loaded model
    """
    return load_data(file_path, 'pickle')


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value (float): Original value
        new_value (float): New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 0 if new_value == 0 else float('inf')
    return ((new_value - old_value) / old_value) * 100


def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average of a time series.
    
    Args:
        data (List[float]): Time series data
        window_size (int): Size of the moving window
        
    Returns:
        List[float]: Moving average values
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    if len(data) < window_size:
        return [np.mean(data)] * len(data)
    
    moving_avg = []
    for i in range(len(data)):
        if i < window_size - 1:
            moving_avg.append(np.mean(data[:i+1]))
        else:
            moving_avg.append(np.mean(data[i-window_size+1:i+1]))
    
    return moving_avg


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to create lags for
        lags (List[int]): List of lag periods
        
    Returns:
        pd.DataFrame: Dataframe with lag features
    """
    df_copy = df.copy()
    
    for lag in lags:
        df_copy[f'{column}_lag_{lag}'] = df_copy[column].shift(lag)
    
    return df_copy


def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to create rolling features for
        windows (List[int]): List of window sizes
        
    Returns:
        pd.DataFrame: Dataframe with rolling features
    """
    df_copy = df.copy()
    
    for window in windows:
        df_copy[f'{column}_rolling_mean_{window}'] = df_copy[column].rolling(window=window).mean()
        df_copy[f'{column}_rolling_std_{window}'] = df_copy[column].rolling(window=window).std()
        df_copy[f'{column}_rolling_min_{window}'] = df_copy[column].rolling(window=window).min()
        df_copy[f'{column}_rolling_max_{window}'] = df_copy[column].rolling(window=window).max()
    
    return df_copy


def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str,
                    title: str = "Time Series Plot", figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot time series data.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data
        date_column (str): Name of the date column
        value_column (str): Name of the value column
        title (str): Title for the plot
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(df[date_column], df[value_column], linewidth=1)
    plt.title(title)
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df (pd.DataFrame): Input dataframe
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with features and importance
        top_n (int): Number of top features to show
        figsize (Tuple[int, int]): Figure size
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Predictions vs Actual",
                              figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Title for the plot
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot: Actual vs Predicted')
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
    """
    print("Model Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("-" * 30)


def create_summary_report(metrics: Dict[str, float], model_name: str,
                         save_path: Optional[str] = None) -> str:
    """
    Create a summary report of model performance.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
        model_name (str): Name of the model
        save_path (str, optional): Path to save the report
        
    Returns:
        str: Summary report text
    """
    report = f"""
Electricity Prediction Model Summary Report
==========================================

Model: {model_name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
-------------------
RMSE (Root Mean Square Error): {metrics.get('rmse', 'N/A'):.4f}
MAE (Mean Absolute Error): {metrics.get('mae', 'N/A'):.4f}
R² (R-squared): {metrics.get('r2', 'N/A'):.4f}
MAPE (Mean Absolute Percentage Error): {metrics.get('mape', 'N/A'):.2f}%

Model Quality Assessment:
------------------------
"""
    
    r2_score = metrics.get('r2', 0)
    if r2_score >= 0.9:
        report += "Excellent model performance (R² ≥ 0.9)\n"
    elif r2_score >= 0.8:
        report += "Good model performance (0.8 ≤ R² < 0.9)\n"
    elif r2_score >= 0.7:
        report += "Fair model performance (0.7 ≤ R² < 0.8)\n"
    else:
        report += "Poor model performance (R² < 0.7)\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Summary report saved to {save_path}")
    
    return report


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return a summary.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Data quality summary
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Check for columns with high missing values
    missing_threshold = 0.5
    high_missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
    quality_report['high_missing_columns'] = high_missing_cols
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    quality_report['constant_columns'] = constant_cols
    
    return quality_report


def setup_logging(log_file: str = 'appliance_energy_prediction.log',
                 log_level: str = 'INFO') -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to log file
        log_level (str): Logging level
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging setup completed")


# Appliance-specific utility functions

def get_appliance_power_ranges() -> Dict[str, Dict[str, int]]:
    """
    Get typical power consumption ranges for different appliances.
    
    Returns:
        Dict[str, Dict[str, int]]: Power ranges by appliance type
    """
    return {
        'Refrigerator': {'min': 100, 'max': 800, 'typical': 400},
        'Air_Conditioner': {'min': 500, 'max': 3000, 'typical': 1500},
        'Washing_Machine': {'min': 300, 'max': 2000, 'typical': 800},
        'Television': {'min': 50, 'max': 400, 'typical': 150},
        'Microwave': {'min': 600, 'max': 1500, 'typical': 1000},
        'Water_Heater': {'min': 1000, 'max': 5000, 'typical': 3000},
        'Dishwasher': {'min': 1200, 'max': 2400, 'typical': 1800},
        'Computer': {'min': 200, 'max': 800, 'typical': 400},
        'Heater': {'min': 500, 'max': 3000, 'typical': 1500},
        'Fan': {'min': 20, 'max': 200, 'typical': 75}
    }


def validate_appliance_power(appliance_type: str, power_rating: float) -> Tuple[bool, str]:
    """
    Validate if power rating is reasonable for appliance type.
    
    Args:
        appliance_type (str): Type of appliance
        power_rating (float): Power rating in watts
        
    Returns:
        Tuple[bool, str]: Validation result and message
    """
    power_ranges = get_appliance_power_ranges()
    
    if appliance_type not in power_ranges:
        return True, "Unknown appliance type, cannot validate power range"
    
    range_info = power_ranges[appliance_type]
    if range_info['min'] <= power_rating <= range_info['max']:
        return True, "Power rating is within expected range"
    else:
        return False, f"Power rating {power_rating}W seems unusual for {appliance_type}. Expected range: {range_info['min']}-{range_info['max']}W"


def calculate_carbon_footprint(daily_kwh: float, grid_carbon_factor: float = 0.82) -> Dict[str, float]:
    """
    Calculate carbon footprint based on energy consumption.
    
    Args:
        daily_kwh (float): Daily energy consumption in kWh
        grid_carbon_factor (float): Grid carbon emission factor (kg CO2/kWh) - India average
        
    Returns:
        Dict[str, float]: Carbon footprint estimates
    """
    daily_co2 = daily_kwh * grid_carbon_factor
    
    return {
        'daily_co2_kg': daily_co2,
        'monthly_co2_kg': daily_co2 * 30,
        'yearly_co2_kg': daily_co2 * 365,
        'yearly_co2_tonnes': daily_co2 * 365 / 1000
    }


def get_energy_efficiency_recommendations(appliance_type: str, age: float, efficiency_rating: float) -> List[str]:
    """
    Get energy efficiency recommendations for an appliance.
    
    Args:
        appliance_type (str): Type of appliance
        age (float): Age of appliance in years
        efficiency_rating (float): Current efficiency rating (1-5)
        
    Returns:
        List[str]: List of recommendations
    """
    recommendations = []
    
    # Age-based recommendations
    if age > 10:
        recommendations.append(f"Consider replacing your {appliance_type.lower().replace('_', ' ')} as it's over 10 years old and likely inefficient")
    elif age > 5:
        recommendations.append(f"Your {appliance_type.lower().replace('_', ' ')} is {age:.0f} years old. Consider upgrading for better efficiency")
    
    # Efficiency-based recommendations
    if efficiency_rating <= 2:
        recommendations.append("Your appliance has a low efficiency rating. Upgrading to a 4-5 star rated appliance could save significant energy")
    elif efficiency_rating == 3:
        recommendations.append("Consider upgrading to a 4-5 star rated appliance for better energy savings")
    
    # Appliance-specific recommendations
    appliance_tips = {
        'Refrigerator': [
            "Keep your refrigerator at 37-40°F (3-4°C) and freezer at 5°F (-15°C)",
            "Clean the coils behind/underneath your refrigerator annually",
            "Don't overfill or underfill your refrigerator"
        ],
        'Air_Conditioner': [
            "Set thermostat to 78°F (26°C) when home, higher when away",
            "Clean or replace filters monthly during peak use",
            "Use ceiling fans to circulate air and feel cooler at higher temperatures"
        ],
        'Washing_Machine': [
            "Wash clothes in cold water when possible",
            "Only run full loads",
            "Clean the lint filter after every load"
        ],
        'Water_Heater': [
            "Set water heater temperature to 120°F (49°C)",
            "Insulate your water heater and hot water pipes",
            "Take shorter showers"
        ],
        'Television': [
            "Turn off TV when not in use",
            "Adjust brightness settings",
            "Enable power saving mode"
        ]
    }
    
    if appliance_type in appliance_tips:
        recommendations.extend(appliance_tips[appliance_type])
    
    return recommendations


def calculate_payback_period(current_cost: float, new_appliance_cost: float, energy_savings: float) -> Dict[str, float]:
    """
    Calculate payback period for appliance upgrade.
    
    Args:
        current_cost (float): Current monthly energy cost
        new_appliance_cost (float): Cost of new appliance
        energy_savings (float): Monthly energy savings from new appliance
        
    Returns:
        Dict[str, float]: Payback analysis
    """
    if energy_savings <= 0:
        return {
            'payback_months': float('inf'),
            'payback_years': float('inf'),
            'annual_savings': energy_savings * 12,
            'recommendation': 'No savings expected - upgrade not recommended'
        }
    
    payback_months = new_appliance_cost / energy_savings
    
    recommendation = ""
    if payback_months <= 12:
        recommendation = "Excellent investment - payback within 1 year"
    elif payback_months <= 24:
        recommendation = "Good investment - payback within 2 years"
    elif payback_months <= 60:
        recommendation = "Reasonable investment - consider upgrade"
    else:
        recommendation = "Long payback period - may not be cost effective"
    
    return {
        'payback_months': payback_months,
        'payback_years': payback_months / 12,
        'annual_savings': energy_savings * 12,
        'recommendation': recommendation
    }


def create_energy_usage_visualization(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create visualizations for energy usage analysis.
    
    Args:
        df (pd.DataFrame): Dataframe containing appliance data
        save_path (str, optional): Path to save the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Energy consumption by appliance type
    if 'appliance_type' in df.columns and 'daily_energy_kwh' in df.columns:
        appliance_energy = df.groupby('appliance_type')['daily_energy_kwh'].mean().sort_values(ascending=False)
        axes[0, 0].bar(appliance_energy.index, appliance_energy.values)
        axes[0, 0].set_title('Average Daily Energy Consumption by Appliance Type')
        axes[0, 0].set_xlabel('Appliance Type')
        axes[0, 0].set_ylabel('Daily Energy (kWh)')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Power rating distribution
    if 'power_rating_watts' in df.columns:
        axes[0, 1].hist(df['power_rating_watts'], bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of Power Ratings')
        axes[0, 1].set_xlabel('Power Rating (Watts)')
        axes[0, 1].set_ylabel('Frequency')
    
    # Usage hours vs energy consumption
    if 'usage_hours_per_day' in df.columns and 'daily_energy_kwh' in df.columns:
        axes[1, 0].scatter(df['usage_hours_per_day'], df['daily_energy_kwh'], alpha=0.6)
        axes[1, 0].set_title('Usage Hours vs Daily Energy Consumption')
        axes[1, 0].set_xlabel('Usage Hours per Day')
        axes[1, 0].set_ylabel('Daily Energy (kWh)')
    
    # Efficiency rating distribution
    if 'efficiency_rating' in df.columns:
        efficiency_counts = df['efficiency_rating'].value_counts().sort_index()
        axes[1, 1].bar(efficiency_counts.index, efficiency_counts.values)
        axes[1, 1].set_title('Distribution of Efficiency Ratings')
        axes[1, 1].set_xlabel('Efficiency Rating (1-5 stars)')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Energy usage visualization saved to {save_path}")
    
    plt.show()


def generate_energy_report(appliance_data: Dict[str, Any], prediction: float) -> Dict[str, Any]:
    """
    Generate a comprehensive energy report for an appliance.
    
    Args:
        appliance_data (Dict[str, Any]): Appliance information
        prediction (float): Predicted daily energy consumption in kWh
        
    Returns:
        Dict[str, Any]: Comprehensive energy report
    """
    # Basic calculations
    daily_cost = prediction * 6.0  # Assuming 6 INR per kWh
    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365
    
    # Carbon footprint
    carbon_footprint = calculate_carbon_footprint(prediction)
    
    # Power validation
    power_valid, power_message = validate_appliance_power(
        appliance_data.get('appliance_type', 'Unknown'),
        appliance_data.get('power_rating_watts', 0)
    )
    
    # Efficiency recommendations
    recommendations = get_energy_efficiency_recommendations(
        appliance_data.get('appliance_type', 'Unknown'),
        appliance_data.get('appliance_age_years', 0),
        appliance_data.get('efficiency_rating', 3)
    )
    
    # Generate report
    report = {
        'appliance_info': appliance_data,
        'energy_consumption': {
            'daily_kwh': prediction,
            'monthly_kwh': prediction * 30,
            'yearly_kwh': prediction * 365
        },
        'cost_analysis': {
            'daily_cost_inr': daily_cost,
            'monthly_cost_inr': monthly_cost,
            'yearly_cost_inr': yearly_cost
        },
        'environmental_impact': carbon_footprint,
        'power_validation': {
            'is_valid': power_valid,
            'message': power_message
        },
        'recommendations': recommendations,
        'generated_at': datetime.now().isoformat()
    }
    
    return report
