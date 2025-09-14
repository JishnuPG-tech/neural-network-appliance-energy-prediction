"""
Data Processing Module for Appliance Energy Prediction

This module contains functions and classes for data preprocessing,
feature engineering, and data transformation specifically designed
for appliance energy consumption prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApplianceDataProcessor:
    """
    A class for processing appliance energy consumption data.
    
    This class provides methods for data cleaning, feature engineering,
    and preprocessing specifically designed for appliance energy prediction.
    """
    
    def __init__(self):
        """Initialize the ApplianceDataProcessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None
        self.categorical_features = ['appliance_type', 'location', 'income_level', 'season', 'usage_pattern']
        self.numerical_features = ['power_rating_watts', 'usage_hours_per_day', 'efficiency_rating', 
                                 'appliance_age_years', 'household_size']
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the appliance dataset by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting appliance data cleaning...")
        
        # Remove duplicates
        original_shape = df.shape
        df = df.drop_duplicates()
        logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values for appliance-specific columns
        if 'power_rating_watts' in df.columns:
            df['power_rating_watts'].fillna(df['power_rating_watts'].median(), inplace=True)
        
        if 'usage_hours_per_day' in df.columns:
            # Cap usage hours at 24
            df['usage_hours_per_day'] = df['usage_hours_per_day'].clip(0, 24)
            df['usage_hours_per_day'].fillna(df['usage_hours_per_day'].median(), inplace=True)
        
        if 'efficiency_rating' in df.columns:
            # Efficiency rating should be between 1-5
            df['efficiency_rating'] = df['efficiency_rating'].clip(1, 5)
            df['efficiency_rating'].fillna(df['efficiency_rating'].median(), inplace=True)
        
        if 'appliance_age_years' in df.columns:
            # Age should be non-negative
            df['appliance_age_years'] = df['appliance_age_years'].clip(0, None)
            df['appliance_age_years'].fillna(df['appliance_age_years'].median(), inplace=True)
        
        if 'household_size' in df.columns:
            # Household size should be positive
            df['household_size'] = df['household_size'].clip(1, None)
            df['household_size'].fillna(df['household_size'].median(), inplace=True)
        
        # Handle categorical columns
        categorical_defaults = {
            'appliance_type': 'Unknown',
            'location': 'Urban',
            'income_level': 'Medium',
            'season': 'Summer',
            'usage_pattern': 'Moderate'
        }
        
        for col, default_value in categorical_defaults.items():
            if col in df.columns:
                df[col].fillna(default_value, inplace=True)
        
        # Remove outliers using IQR method for numerical columns
        for col in self.numerical_features:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_removed = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                if outliers_removed > 0:
                    logger.info(f"Removed {outliers_removed} outliers from {col}")
        
        logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create appliance-specific features from existing data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new features
        """
        logger.info("Starting appliance feature engineering...")
        
        # Basic energy calculation features
        if 'power_rating_watts' in df.columns and 'usage_hours_per_day' in df.columns:
            df['daily_energy_wh'] = df['power_rating_watts'] * df['usage_hours_per_day']
            df['daily_energy_kwh'] = df['daily_energy_wh'] / 1000
            df['monthly_energy_kwh'] = df['daily_energy_kwh'] * 30
            df['yearly_energy_kwh'] = df['daily_energy_kwh'] * 365
        
        # Efficiency features
        if 'efficiency_rating' in df.columns:
            df['efficiency_factor'] = 6 - df['efficiency_rating']  # Higher rating = lower factor
            if 'power_rating_watts' in df.columns:
                df['power_efficiency_ratio'] = df['power_rating_watts'] / df['efficiency_rating']
        
        # Age-related features
        if 'appliance_age_years' in df.columns:
            df['age_category'] = pd.cut(df['appliance_age_years'], 
                                      bins=[0, 2, 5, 10, float('inf')],
                                      labels=['New', 'Recent', 'Moderate', 'Old'])
            
            # Efficiency degradation with age
            if 'efficiency_rating' in df.columns:
                df['age_efficiency_impact'] = df['appliance_age_years'] * (6 - df['efficiency_rating'])
        
        # Usage intensity features
        if 'usage_hours_per_day' in df.columns:
            df['usage_intensity'] = pd.cut(df['usage_hours_per_day'],
                                         bins=[0, 2, 6, 12, 24],
                                         labels=['Light', 'Moderate', 'Heavy', 'Continuous'])
            
            df['weekend_usage_factor'] = np.where(df['usage_hours_per_day'] > 8, 1.2, 1.0)
        
        # Household features
        if 'household_size' in df.columns:
            df['household_category'] = pd.cut(df['household_size'],
                                            bins=[0, 2, 4, 6, float('inf')],
                                            labels=['Small', 'Medium', 'Large', 'XLarge'])
            
            if 'usage_hours_per_day' in df.columns:
                df['usage_per_person'] = df['usage_hours_per_day'] / df['household_size']
        
        # Seasonal adjustments
        if 'season' in df.columns:
            season_multipliers = {
                'Summer': 1.3,  # Higher usage for cooling
                'Winter': 1.2,  # Higher usage for heating
                'Spring': 1.0,
                'Autumn': 0.9
            }
            df['seasonal_multiplier'] = df['season'].map(season_multipliers).fillna(1.0)
            
            if 'daily_energy_kwh' in df.columns:
                df['seasonal_adjusted_energy'] = df['daily_energy_kwh'] * df['seasonal_multiplier']
        
        # Location-based features
        if 'location' in df.columns:
            location_factors = {
                'Urban': 1.1,    # Higher consumption in urban areas
                'Suburban': 1.0,
                'Rural': 0.9     # Lower consumption in rural areas
            }
            df['location_factor'] = df['location'].map(location_factors).fillna(1.0)
        
        # Income-based features
        if 'income_level' in df.columns:
            income_factors = {
                'High': 1.2,     # Higher consumption for high income
                'Medium': 1.0,
                'Low': 0.8       # Lower consumption for low income
            }
            df['income_factor'] = df['income_level'].map(income_factors).fillna(1.0)
        
        # Usage pattern features
        if 'usage_pattern' in df.columns:
            pattern_factors = {
                'Heavy': 1.4,
                'Moderate': 1.0,
                'Light': 0.6,
                'Occasional': 0.3
            }
            df['pattern_factor'] = df['usage_pattern'].map(pattern_factors).fillna(1.0)
        
        # Interaction features
        if 'power_rating_watts' in df.columns and 'household_size' in df.columns:
            df['power_per_person'] = df['power_rating_watts'] / df['household_size']
        
        if 'appliance_age_years' in df.columns and 'usage_hours_per_day' in df.columns:
            df['age_usage_interaction'] = df['appliance_age_years'] * df['usage_hours_per_day']
        
        # Energy cost estimation (assuming 6 INR per kWh)
        if 'daily_energy_kwh' in df.columns:
            df['daily_cost_inr'] = df['daily_energy_kwh'] * 6
            df['monthly_cost_inr'] = df['daily_cost_inr'] * 30
            df['yearly_cost_inr'] = df['daily_cost_inr'] * 365
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables for appliance data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the encoder (True for training, False for prediction)
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Handle categorical features with one-hot encoding
        categorical_cols_present = [col for col in self.categorical_features if col in df.columns]
        
        if categorical_cols_present:
            if fit:
                # Fit and transform for training data
                encoded_features = self.one_hot_encoder.fit_transform(df[categorical_cols_present])
                feature_names = self.one_hot_encoder.get_feature_names_out(categorical_cols_present)
            else:
                # Transform only for prediction data
                encoded_features = self.one_hot_encoder.transform(df[categorical_cols_present])
                feature_names = self.one_hot_encoder.get_feature_names_out(categorical_cols_present)
            
            # Create dataframe with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            
            # Drop original categorical columns and add encoded ones
            df_encoded = df_encoded.drop(columns=categorical_cols_present)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        # Handle any remaining categorical columns with label encoding
        remaining_categorical = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in remaining_categorical:
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df_encoded[col].astype(str))
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        logger.warning(f"Unseen categories in {col}: {unseen_values}")
                        # Replace unseen values with the most frequent category
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_encoded[col] = df_encoded[col].replace(list(unseen_values), most_frequent)
                    
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                else:
                    logger.warning(f"Label encoder for {col} not found. Skipping.")
        
        logger.info(f"Categorical encoding completed. Shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            Tuple: Scaled training and test features
        """
        logger.info("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data...")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split completed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def process_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> dict:
        """
        Complete data processing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Processed data dictionary
        """
        logger.info("Starting complete data processing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Create features
        df_features = self.create_features(df_clean)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_features)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df_encoded, target_column, test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        
        logger.info("Data processing pipeline completed successfully.")
        return processed_data


def remove_outliers(df: pd.DataFrame, columns: list, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to check for outliers
        method (str): Method to use ('iqr' or 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    logger.info(f"Removing outliers using {method} method...")
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean = df_clean[z_scores < 3]
    
    logger.info(f"Outliers removed. Original shape: {df.shape}, New shape: {df_clean.shape}")
    return df_clean


def create_time_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        datetime_column (str): Name of the datetime column
        
    Returns:
        pd.DataFrame: Dataframe with time features
    """
    logger.info("Creating time-based features...")
    
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Extract time components
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['hour'] = df[datetime_column].dt.hour
    df['dayofweek'] = df[datetime_column].dt.dayofweek
    df['quarter'] = df[datetime_column].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    logger.info("Time-based features created successfully.")
    return df


def prepare_appliance_features(appliance_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare features for a single appliance prediction.
    
    Args:
        appliance_data (Dict[str, Any]): Dictionary containing appliance information
        
    Returns:
        pd.DataFrame: Prepared features ready for model prediction
    """
    # Create a dataframe from the input data
    df = pd.DataFrame([appliance_data])
    
    # Initialize processor
    processor = ApplianceDataProcessor()
    
    # Create features
    df_features = processor.create_features(df)
    
    # Handle categorical encoding for prediction
    df_encoded = processor.encode_categorical(df_features, fit=False)
    
    return df_encoded


def validate_appliance_data(appliance_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate appliance data for prediction.
    
    Args:
        appliance_data (Dict[str, Any]): Dictionary containing appliance information
        
    Returns:
        Tuple[bool, List[str]]: Validation status and list of errors
    """
    errors = []
    
    # Required fields
    required_fields = ['appliance_type', 'power_rating_watts', 'usage_hours_per_day', 
                      'efficiency_rating', 'appliance_age_years', 'household_size']
    
    for field in required_fields:
        if field not in appliance_data or appliance_data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate data ranges
    if 'power_rating_watts' in appliance_data:
        if not 0 < appliance_data['power_rating_watts'] <= 10000:
            errors.append("Power rating must be between 1 and 10000 watts")
    
    if 'usage_hours_per_day' in appliance_data:
        if not 0 <= appliance_data['usage_hours_per_day'] <= 24:
            errors.append("Usage hours must be between 0 and 24")
    
    if 'efficiency_rating' in appliance_data:
        if not 1 <= appliance_data['efficiency_rating'] <= 5:
            errors.append("Efficiency rating must be between 1 and 5")
    
    if 'appliance_age_years' in appliance_data:
        if not 0 <= appliance_data['appliance_age_years'] <= 50:
            errors.append("Appliance age must be between 0 and 50 years")
    
    if 'household_size' in appliance_data:
        if not 1 <= appliance_data['household_size'] <= 20:
            errors.append("Household size must be between 1 and 20")
    
    # Validate categorical values
    valid_appliance_types = ['Refrigerator', 'Air_Conditioner', 'Washing_Machine', 'Television', 
                           'Microwave', 'Water_Heater', 'Dishwasher', 'Computer', 'Heater', 'Fan']
    if 'appliance_type' in appliance_data:
        if appliance_data['appliance_type'] not in valid_appliance_types:
            errors.append(f"Invalid appliance type. Must be one of: {valid_appliance_types}")
    
    valid_locations = ['Urban', 'Suburban', 'Rural']
    if 'location' in appliance_data:
        if appliance_data['location'] not in valid_locations:
            errors.append(f"Invalid location. Must be one of: {valid_locations}")
    
    valid_income_levels = ['Low', 'Medium', 'High']
    if 'income_level' in appliance_data:
        if appliance_data['income_level'] not in valid_income_levels:
            errors.append(f"Invalid income level. Must be one of: {valid_income_levels}")
    
    valid_seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    if 'season' in appliance_data:
        if appliance_data['season'] not in valid_seasons:
            errors.append(f"Invalid season. Must be one of: {valid_seasons}")
    
    valid_usage_patterns = ['Light', 'Moderate', 'Heavy', 'Occasional']
    if 'usage_pattern' in appliance_data:
        if appliance_data['usage_pattern'] not in valid_usage_patterns:
            errors.append(f"Invalid usage pattern. Must be one of: {valid_usage_patterns}")
    
    return len(errors) == 0, errors


def get_appliance_defaults() -> Dict[str, Any]:
    """
    Get default values for appliance data.
    
    Returns:
        Dict[str, Any]: Default appliance data
    """
    return {
        'appliance_type': 'Refrigerator',
        'power_rating_watts': 150,
        'usage_hours_per_day': 24,
        'efficiency_rating': 3,
        'appliance_age_years': 2,
        'household_size': 4,
        'location': 'Urban',
        'income_level': 'Medium',
        'season': 'Summer',
        'usage_pattern': 'Moderate'
    }


def estimate_energy_cost(daily_kwh: float, electricity_rate: float = 6.0) -> Dict[str, float]:
    """
    Estimate energy costs based on consumption.
    
    Args:
        daily_kwh (float): Daily energy consumption in kWh
        electricity_rate (float): Electricity rate in INR per kWh
        
    Returns:
        Dict[str, float]: Cost estimates
    """
    return {
        'daily_cost': daily_kwh * electricity_rate,
        'monthly_cost': daily_kwh * electricity_rate * 30,
        'yearly_cost': daily_kwh * electricity_rate * 365
    }
