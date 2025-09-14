"""
Flask Web Application for Appliance Energy Prediction

This module creates a web interface for the neural network-based appliance energy 
prediction model, allowing users to input appliance details and get energy 
consumption predictions through a beautiful web interface.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'appliance_energy_prediction_secret_key_2024'

# Global variables
model = None
scaler = None
feature_names = None
model_info = {}


def load_trained_model():
    """Load the trained neural network model and preprocessing components."""
    global model, scaler, feature_names, model_info
    
    try:
        # Load the neural network model
        model_path = os.path.join('..', 'models', 'appliance_energy_predictor.h5')
        if os.path.exists(model_path):
            model = tf_load_model(model_path)
            logger.info("✅ Neural network model loaded successfully")
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return False
            
        # Load the feature scaler
        scaler_path = os.path.join('..', 'models', 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("✅ Feature scaler loaded successfully")
        else:
            logger.error(f"❌ Scaler file not found: {scaler_path}")
            return False
            
        # Load feature names
        feature_names_path = os.path.join('..', 'models', 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            logger.info(f"✅ Feature names loaded ({len(feature_names)} features)")
        else:
            logger.error(f"❌ Feature names file not found: {feature_names_path}")
            return False
            
        # Load model metadata
        model_info_path = os.path.join('..', 'models', 'model_info.pkl')
        if os.path.exists(model_info_path):
            model_info = joblib.load(model_info_path)
            logger.info("✅ Model metadata loaded successfully")
        else:
            logger.warning("⚠️ Model metadata not found, using defaults")
            model_info = {
                'model_type': 'Neural Network (TensorFlow/Keras)',
                'target_variable': 'daily_consumption_kwh',
                'n_features': len(feature_names) if feature_names else 'Unknown'
            }
            
        logger.info("🎉 All model components loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model components: {str(e)}")
        model_info = {
            'status': f'Error loading model: {str(e)}',
            'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return False


def prepare_features(appliance_data):
    """Prepare features for the neural network model."""
    try:
        # Define all possible categories for encoding
        appliance_types = ['Air Conditioner', 'Refrigerator', 'Television', 'Washing Machine', 
                          'Water Heater', 'Microwave', 'Ceiling Fan', 'LED Lights', 
                          'Laptop/Computer', 'Dishwasher']
        locations = ['Urban', 'Suburban', 'Rural']
        income_levels = ['Low', 'Medium', 'High']
        seasons = ['Summer', 'Monsoon', 'Winter', 'Spring']
        usage_patterns = ['Light', 'Moderate', 'Heavy']
        
        # Create feature vector
        features = {}
        
        # Numerical features
        features['power_rating_watts'] = float(appliance_data.get('power_rating', 1000))
        features['usage_hours_per_day'] = float(appliance_data.get('usage_hours', 8))
        features['efficiency_rating'] = float(appliance_data.get('efficiency_rating', 3))
        features['appliance_age_years'] = float(appliance_data.get('appliance_age', 2))
        features['household_size'] = float(appliance_data.get('household_size', 4))
        
        # One-hot encode categorical features
        for appliance in appliance_types:
            features[f'appliance_type_{appliance}'] = 1 if appliance_data.get('appliance_type') == appliance else 0
            
        for location in locations:
            features[f'location_{location}'] = 1 if appliance_data.get('location') == location else 0
            
        for income in income_levels:
            features[f'income_level_{income}'] = 1 if appliance_data.get('income_level') == income else 0
            
        for season in seasons:
            features[f'season_{season}'] = 1 if appliance_data.get('season') == season else 0
            
        for pattern in usage_patterns:
            features[f'usage_pattern_{pattern}'] = 1 if appliance_data.get('usage_pattern') == pattern else 0
        
        # Create DataFrame and reindex to match training features
        df = pd.DataFrame([features])
        df = df.reindex(columns=feature_names, fill_value=0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None


@app.route('/')
def home():
    """Home page with appliance energy prediction interface."""
    return render_template('index.html', 
                         model_info=model_info,
                         page_title="Appliance Energy Predictor")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle appliance energy consumption predictions."""
    if request.method == 'GET':
        # Show the prediction form
        return render_template('predict.html', 
                             model_info=model_info,
                             page_title="Predict Energy Consumption")
    
    try:
        # Get form data
        appliance_data = {
            'appliance_type': request.form.get('appliance_type'),
            'power_rating': request.form.get('power_rating'),
            'usage_hours': request.form.get('usage_hours'),
            'efficiency_rating': request.form.get('efficiency_rating'),
            'appliance_age': request.form.get('appliance_age'),
            'household_size': request.form.get('household_size'),
            'location': request.form.get('location'),
            'income_level': request.form.get('income_level'),
            'season': request.form.get('season'),
            'usage_pattern': request.form.get('usage_pattern')
        }
        
        # Validate required fields
        required_fields = ['appliance_type', 'power_rating', 'usage_hours', 
                          'efficiency_rating', 'appliance_age', 'household_size']
        missing_fields = [field for field in required_fields if not appliance_data.get(field)]
        
        if missing_fields:
            flash(f"Missing required fields: {', '.join(missing_fields)}", 'error')
            return render_template('predict.html', model_info=model_info)
        
        # Check if model is loaded
        if model is None:
            flash("Model not loaded. Please check model files.", 'error')
            return render_template('predict.html', model_info=model_info)
        
        # Prepare features
        features_df = prepare_features(appliance_data)
        if features_df is None:
            flash("Error preparing features for prediction.", 'error')
            return render_template('predict.html', model_info=model_info)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        
        # Calculate additional insights
        daily_consumption = max(0, prediction)  # Ensure non-negative
        monthly_consumption = daily_consumption * 30
        monthly_cost = monthly_consumption * 6  # ₹6 per kWh average in India
        annual_cost = monthly_cost * 12
        
        # Prepare result
        result = {
            'daily_consumption': round(daily_consumption, 3),
            'monthly_consumption': round(monthly_consumption, 2),
            'monthly_cost': round(monthly_cost, 2),
            'annual_cost': round(annual_cost, 2),
            'appliance_details': appliance_data,
            'prediction_confidence': 'High' if daily_consumption > 0.1 else 'Low'
        }
        
        return render_template('predict.html', 
                             model_info=model_info,
                             result=result,
                             appliance_data=appliance_data)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        flash(f"Prediction error: {str(e)}", 'error')
        return render_template('predict.html', model_info=model_info)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for energy consumption predictions."""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Prepare features
        features_df = prepare_features(data)
        if features_df is None:
            return jsonify({'error': 'Error preparing features'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)[0][0]
        daily_consumption = max(0, prediction)
        
        # Calculate costs
        monthly_consumption = daily_consumption * 30
        monthly_cost = monthly_consumption * 6
        annual_cost = monthly_cost * 12
        
        result = {
            'daily_consumption_kwh': round(daily_consumption, 3),
            'monthly_consumption_kwh': round(monthly_consumption, 2),
            'monthly_cost_inr': round(monthly_cost, 2),
            'annual_cost_inr': round(annual_cost, 2),
            'model_info': {
                'model_type': model_info.get('model_type', 'Neural Network'),
                'prediction_timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    if request.method == 'GET':
        return render_template('predict.html', 
                             feature_names=feature_names,
                             model_info=model_info)
    
    elif request.method == 'POST':
        try:
            if model is None:
                flash('Model not loaded. Please check model status.', 'error')
                return redirect(url_for('predict'))
            
            # Get form data
            form_data = request.form.to_dict()
            
            # Convert form data to appropriate format
            input_data = []
            for feature in feature_names or []:
                value = form_data.get(feature, 0)
                try:
                    input_data.append(float(value))
                except ValueError:
                    input_data.append(0.0)
            
            if not input_data:
                flash('No valid input data provided', 'error')
                return redirect(url_for('predict'))
            
            # Make prediction
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            
            # Prepare response
            result = {
                'prediction': float(prediction),
                'input_data': dict(zip(feature_names or [], input_data)),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            flash(f'Error during prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 400
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Extract features
        if isinstance(data, dict) and 'features' in data:
            features = data['features']
        elif isinstance(data, list):
            features = data
        else:
            return jsonify({
                'error': 'Invalid data format. Expected list of features or dict with "features" key',
                'status': 'error'
            }), 400
        
        # Convert to numpy array
        input_array = np.array(features).reshape(1, -1)
        
        # Validate input dimensions
        if feature_names and len(features) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(features)}',
                'status': 'error'
            }), 400
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Return result
        return jsonify({
            'prediction': float(prediction),
            'input_features': features,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/model_info')
def api_model_info():
    """API endpoint for model information."""
    return jsonify(model_info)


@app.route('/api/features')
def api_features():
    """API endpoint for feature names."""
    return jsonify({
        'features': feature_names or [],
        'count': len(feature_names) if feature_names else 0
    })


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """File upload route for batch predictions."""
    if request.method == 'GET':
        return render_template('upload.html', model_info=model_info)
    
@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    """Handle batch predictions from CSV file."""
    if request.method == 'GET':
        return render_template('batch.html', 
                             model_info=model_info,
                             page_title="Batch Predictions")
    
    try:
        if model is None:
            flash('Model not loaded. Please check model status.', 'error')
            return redirect(url_for('batch_predict'))
        
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('batch_predict'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('batch_predict'))
        
        if file and file.filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file)
            
            # Validate required columns
            required_columns = ['appliance_type', 'power_rating_watts', 'usage_hours_per_day', 
                              'efficiency_rating', 'appliance_age_years', 'household_size']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                flash(f'Missing required columns: {", ".join(missing_columns)}', 'error')
                return redirect(url_for('batch_predict'))
            
            # Prepare features for each row
            predictions = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    appliance_data = {
                        'appliance_type': row.get('appliance_type'),
                        'power_rating': row.get('power_rating_watts', row.get('power_rating')),
                        'usage_hours': row.get('usage_hours_per_day', row.get('usage_hours')),
                        'efficiency_rating': row.get('efficiency_rating'),
                        'appliance_age': row.get('appliance_age_years', row.get('appliance_age')),
                        'household_size': row.get('household_size'),
                        'location': row.get('location', 'Urban'),
                        'income_level': row.get('income_level', 'Medium'),
                        'season': row.get('season', 'Summer'),
                        'usage_pattern': row.get('usage_pattern', 'Moderate')
                    }
                    
                    features_df = prepare_features(appliance_data)
                    if features_df is not None:
                        features_scaled = scaler.transform(features_df)
                        prediction = model.predict(features_scaled, verbose=0)[0][0]
                        predictions.append(max(0, prediction))
                    else:
                        predictions.append(None)
                        errors.append(f"Row {idx}: Error preparing features")
                        
                except Exception as e:
                    predictions.append(None)
                    errors.append(f"Row {idx}: {str(e)}")
            
            # Add predictions to dataframe
            df['predicted_daily_consumption_kwh'] = predictions
            df['predicted_monthly_cost_inr'] = [p * 30 * 6 if p else None for p in predictions]
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'appliance_predictions_{timestamp}.csv'
            output_path = os.path.join('..', 'data', 'processed', output_filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            successful_predictions = len([p for p in predictions if p is not None])
            flash(f'Batch predictions complete! {successful_predictions}/{len(df)} successful predictions saved to {output_filename}', 'success')
            
            if errors:
                flash(f'Errors encountered: {"; ".join(errors[:5])}', 'warning')
            
            # Show sample results
            sample_results = df.head(10).to_dict('records')
            
            return render_template('batch_results.html', 
                                 results=sample_results,
                                 total_predictions=len(predictions),
                                 successful_predictions=successful_predictions,
                                 output_filename=output_filename,
                                 model_info=model_info)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(url_for('batch_predict'))
            
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        flash(f'Error processing batch predictions: {str(e)}', 'error')
        return redirect(url_for('batch_predict'))


@app.route('/model_status')
def model_status():
    """Model status page."""
    return render_template('model_status.html', model_info=model_info)


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error"), 500


def create_templates():
    """Create basic HTML templates if they don't exist."""
    templates_dir = 'templates'
    
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info("Created templates directory")
    
    # Basic index template
    index_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Electricity Prediction App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 20px; text-decoration: none; color: #007bff; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Electricity Prediction Application</h1>
        <div class="nav">
            <a href="/">Home</a>
            <a href="/predict">Make Prediction</a>
            <a href="/upload">Batch Predictions</a>
            <a href="/model_status">Model Status</a>
        </div>
        
        <h2>Welcome</h2>
        <p>This application provides electricity consumption predictions using machine learning.</p>
        
        <div class="status">
            <strong>Model Status:</strong> {{ model_info.get('status', 'Unknown') }}
        </div>
        
        <h3>Features:</h3>
        <ul>
            <li>Single prediction with manual input</li>
            <li>Batch predictions from CSV files</li>
            <li>REST API for programmatic access</li>
            <li>Model status monitoring</li>
        </ul>
    </div>
</body>
</html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_template)


if __name__ == '__main__':
    # Create templates if they don't exist
    create_templates()
    
    # Load the trained model
    load_trained_model()
    
    print("🏠 Neural Network-Based Appliance Energy Prediction System")
    print("=" * 60)
    
    if model:
        print(f"✅ Model loaded successfully")
        print(f"🔧 Total features: {len(feature_names) if feature_names else 'Unknown'}")
    else:
        print("❌ Model failed to load")
    
    print("\n🌐 Starting Flask application...")
    print("📱 Web interface available at: http://127.0.0.1:5000")
    print("💡 Use this tool to predict appliance energy consumption")
    print("\nPress Ctrl+C to stop the application")
    print("=" * 60)
    
    # Run the Flask app
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
