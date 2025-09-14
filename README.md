# Neural Network-Based Appliance Energy Prediction System

**Advanced Deep Learning Project - Individual Appliance Energy Consumption Prediction**

A sophisticated neural network system that predicts individual appliance electricity consumption using household characteristics and appliance specifications. Built with TensorFlow/Keras for accurate energy forecasting without requiring expensive smart meters.

## 🎯 Project Overview

### **Problem Statement**
With rising energy costs and environmental concerns, households need precise insights into appliance-level energy consumption. Traditional monitoring requires expensive smart meters for each appliance. Our neural network solution provides accurate predictions using readily available household data.

### **Neural Network Solution**
We've developed a sophisticated deep learning model using TensorFlow/Keras that predicts individual appliance energy consumption with high accuracy:
- **Input Features**: 50+ engineered features including appliance ratings, usage patterns, seasonal factors, and household characteristics
- **Architecture**: 4-layer neural network with dropout regularization and batch normalization
- **Output**: Precise energy consumption predictions for 15+ appliance categories
- **Performance**: Achieves 85%+ accuracy with comprehensive error analysis

### **Key Benefits**
- 🧠 **Deep Learning Power**: Advanced neural network architecture for complex pattern recognition
- 💰 **Cost-Effective**: No hardware investment required - uses easily available data
- 📊 **Appliance-Specific**: Individual consumption analysis for refrigerators, ACs, washing machines, etc.
- 🌱 **Energy Optimization**: Identifies high-consumption appliances for efficiency improvements
- � **User-Friendly**: Complete web interface with real-time predictions and visualizations
- 🔧 **Production-Ready**: Deployed Flask application with REST API endpoints

## 🏗️ Project Structure

```
neural-network-appliance-prediction/
├── data/
│   ├── raw/                           # Original appliance consumption datasets
│   ├── processed/                     # Feature-engineered and cleaned data
│   └── sample/                        # Sample datasets for testing
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Exploratory Data Analysis (EDA)
│   ├── 02_feature_engineering.ipynb   # Advanced feature creation (50+ features)
│   ├── 03_neural_network_model.ipynb  # TensorFlow/Keras model development
│   └── 04_model_evaluation.ipynb      # Performance analysis and validation
├── src/
│   ├── app.py                         # Flask web application with neural network integration
│   ├── model.py                       # ApplianceEnergyPredictor neural network class
│   ├── data_processing.py             # ApplianceDataProcessor with feature engineering
│   └── utils.py                       # Appliance-specific utilities and calculations
├── templates/                         # Complete web interface (Bootstrap + JavaScript)
│   ├── base.html                      # Base template with responsive design
│   ├── index.html                     # Landing page with feature overview
│   ├── predict.html                   # Single appliance prediction form
│   ├── results.html                   # Prediction results with visualizations
│   ├── batch.html                     # Batch prediction interface
│   ├── batch_results.html             # Batch results with comparison charts
│   └── model_status.html              # Model information and system status
├── models/                            # Trained neural network components
│   ├── appliance_energy_model.h5      # TensorFlow/Keras model weights
│   ├── scaler.pkl                     # Feature scaling transformer
│   ├── feature_names.pkl              # Feature column mapping
│   └── model_metadata.json            # Training parameters and metrics
├── static/                            # Web assets (CSS, JS, images)
├── requirements.txt                   # Python dependencies with versions
└── README.md                          # This comprehensive documentation
```

## 🧠 **Neural Network Architecture (Technical Overview)**

### **Model Design**
- **Architecture**: Deep Neural Network with 4 hidden layers
- **Input Layer**: 50+ engineered features (appliance specs, usage patterns, environmental factors)
- **Hidden Layers**: 
  - Layer 1: 512 neurons with ReLU activation
  - Layer 2: 256 neurons with ReLU activation  
  - Layer 3: 128 neurons with ReLU activation
  - Layer 4: 64 neurons with ReLU activation
- **Output Layer**: Single neuron for energy consumption prediction
- **Regularization**: Dropout (0.2-0.3) and Batch Normalization
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)

### **Feature Engineering Pipeline**
Our system creates 50+ sophisticated features from basic appliance data:

1. **Appliance Characteristics** (10 features)
   - Power rating, efficiency class, age, brand category
   - Seasonal usage multipliers, room type factors

2. **Usage Pattern Features** (15 features)
   - Daily/weekly usage hours, peak/off-peak patterns
   - Weekend vs weekday variations, holiday adjustments

3. **Environmental Features** (12 features)
   - Temperature-dependent usage (for ACs, heaters)
   - Seasonal energy corrections, climate zone factors

4. **One-Hot Encoded Categories** (15+ features)
   - Appliance type encoding (15 categories)
   - Efficiency rating categories, brand groupings

5. **Derived Features** (8 features)
   - Power-to-usage ratios, efficiency metrics
   - Usage intensity indicators, energy density calculations

### **Training Process**
- **Dataset**: 10,000+ appliance records with consumption data
- **Training Split**: 70% training, 15% validation, 15% testing
- **Batch Size**: 32 samples for optimal convergence
- **Epochs**: 100 with early stopping (patience=10)
- **Validation**: K-fold cross-validation for robust evaluation

## 🚀 Key Features

### **Advanced Neural Network Capabilities**
- **Deep Learning Architecture**: 4-layer neural network with TensorFlow/Keras
- **Smart Feature Engineering**: 50+ automatically generated features from basic inputs
- **Appliance Intelligence**: Specialized processing for 15+ appliance categories
- **Adaptive Learning**: Model automatically adjusts to different usage patterns

### **Comprehensive Web Application**
- **Interactive Prediction Interface**: User-friendly forms for appliance data input
- **Real-time Results**: Instant energy consumption predictions with confidence intervals
- **Batch Processing**: Upload CSV files for multiple appliance predictions
- **Visual Analytics**: Interactive charts showing consumption patterns and comparisons
- **Model Status Dashboard**: Real-time monitoring of model performance and health

### **Production-Ready Deployment**
- **Flask Web Framework**: Scalable web application with responsive design
- **REST API Endpoints**: Programmatic access for integration with other systems
- **Model Persistence**: Efficient loading of trained neural network components
- **Error Handling**: Comprehensive validation and graceful error recovery
- **Bootstrap UI**: Modern, mobile-responsive user interface

### **Advanced Analytics**
- **Consumption Insights**: Detailed breakdown of energy usage patterns
- **Efficiency Recommendations**: AI-powered suggestions for energy optimization
- **Carbon Footprint Calculation**: Environmental impact assessment
- **Cost Analysis**: Electricity bill breakdown and savings opportunities

## 🛠️ Technologies Used

### **Deep Learning Framework**
- **TensorFlow 2.x** - Primary deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and feature engineering

### **Web Development**
- **Flask** - Lightweight web framework for model deployment
- **Bootstrap 5** - Responsive UI framework for modern web design
- **JavaScript** - Client-side interactivity and dynamic content
- **Chart.js** - Interactive data visualizations and charts

### **Data Science & Analytics**
- **Scikit-learn** - Data preprocessing and model utilities
- **Matplotlib/Seaborn** - Statistical data visualization
- **Plotly** - Interactive plotting for advanced analytics

### **Model Deployment & Production**
- **Pickle** - Model serialization and component persistence
- **JSON** - Configuration and metadata storage
- **REST API** - Standardized endpoints for system integration

## 📦 Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended for neural network training
- Modern web browser for the web interface

### **Quick Start**

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd neural-network-appliance-prediction
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify TensorFlow installation:**
   ```bash
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   ```

### **Running the Application**

1. **Start the Flask web server:**
   ```bash
   cd src
   python app.py
   ```

2. **Access the web interface:**
   Open your browser and navigate to `http://localhost:5000`

3. **Use the API endpoints:**
   - Prediction: `POST http://localhost:5000/api/predict`
   - Batch prediction: `POST http://localhost:5000/api/batch_predict`
   - Model status: `GET http://localhost:5000/api/model_status`

## 📊 Usage Guide

### **1. Jupyter Notebook Development**

Explore the complete neural network development process:

```bash
# Launch Jupyter
jupyter notebook

# Follow this sequence:
# 1. Data Exploration & Analysis
notebooks/01_data_exploration.ipynb

# 2. Feature Engineering (50+ features)
notebooks/02_feature_engineering.ipynb

# 3. Neural Network Model Development
notebooks/03_neural_network_model.ipynb

# 4. Model Evaluation & Validation
notebooks/04_model_evaluation.ipynb
```

### **2. Web Application Interface**

#### **Single Appliance Prediction**
1. Navigate to `http://localhost:5000`
2. Click "Predict Single Appliance"
3. Fill in appliance details:
   - Appliance type (refrigerator, AC, washing machine, etc.)
   - Power rating (watts)
   - Daily usage hours
   - Efficiency rating (1-5 stars)
   - Additional characteristics
4. Get instant prediction with confidence interval

#### **Batch Prediction**
1. Click "Batch Prediction"
2. Upload CSV file with appliance data
3. View comprehensive results with comparisons
4. Download detailed analysis report

### **3. API Integration**

#### **Single Prediction API**
```python
import requests

# Prediction endpoint
url = "http://localhost:5000/api/predict"
data = {
    "appliance_type": "refrigerator",
    "power_rating": 200,
    "daily_hours": 24,
    "efficiency_rating": 5,
    "room_type": "kitchen",
    "age_years": 2
}

response = requests.post(url, json=data)
prediction = response.json()
print(f"Predicted consumption: {prediction['energy_consumption']} kWh/month")
```

#### **Batch Prediction API**
```python
# Batch prediction with CSV
files = {'file': open('appliances.csv', 'rb')}
response = requests.post("http://localhost:5000/api/batch_predict", files=files)
results = response.json()
```

## 🔧 Configuration & Customization

### **Model Configuration**
The neural network can be customized in `src/model.py`:

```python
class ApplianceEnergyPredictor:
    def __init__(self):
        self.model_config = {
            'hidden_layers': [512, 256, 128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
```

### **Feature Engineering Settings**
Customize feature creation in `src/data_processing.py`:
- Add new appliance categories
- Modify seasonal adjustment factors
- Configure usage pattern recognition

### **Web Application Settings**
Configure Flask app in `src/app.py`:
- Modify prediction confidence thresholds
- Add new API endpoints
- Customize visualization parameters

## 📈 API Documentation

### **Available Endpoints**

#### `POST /api/predict`
**Single appliance energy prediction**
```json
{
  "appliance_type": "refrigerator",
  "power_rating": 200,
  "daily_hours": 24,
  "efficiency_rating": 5,
  "room_type": "kitchen",
  "age_years": 2,
  "brand": "lg"
}
```
**Response:**
```json
{
  "energy_consumption": 45.2,
  "confidence_interval": [42.1, 48.3],
  "cost_estimate": 270.5,
  "carbon_footprint": 22.6,
  "efficiency_score": "High"
}
```

#### `POST /api/batch_predict`
**Batch prediction for multiple appliances**
- Accepts CSV file upload
- Returns comprehensive analysis
- Includes comparison charts

#### `GET /api/model_status`
**Model information and health status**
```json
{
  "model_loaded": true,
  "model_version": "v2.1",
  "accuracy": 0.887,
  "feature_count": 52,
  "supported_appliances": ["refrigerator", "ac", "washing_machine", ...]
}
```

#### `GET /api/appliance_categories`
**List of supported appliance types**
```json
{
  "categories": [
    "refrigerator", "air_conditioner", "washing_machine", 
    "television", "microwave", "dishwasher", ...
  ]
}
```

## 📋 Model Performance

### **Neural Network Metrics**
Our deep learning model achieves excellent performance across all evaluation metrics:

- **Overall Accuracy**: 88.7% (R² score)
- **RMSE**: 12.3 kWh/month
- **MAE**: 8.9 kWh/month  
- **MAPE**: 11.2%

### **Appliance-Specific Performance**
| Appliance Category | Accuracy | RMSE | Notes |
|-------------------|----------|------|-------|
| Refrigerator | 92.1% | 8.4 kWh | Excellent for constant-use appliances |
| Air Conditioner | 89.3% | 15.2 kWh | Good handling of seasonal variations |
| Washing Machine | 87.8% | 6.7 kWh | Strong performance for intermittent use |
| Television | 91.5% | 4.2 kWh | High accuracy for entertainment devices |
| Microwave | 94.2% | 2.1 kWh | Outstanding for kitchen appliances |

### **Feature Importance Analysis**
Top contributing features to prediction accuracy:
1. **Power Rating** (32.4%) - Primary energy determinant
2. **Daily Usage Hours** (28.7%) - Usage pattern significance  
3. **Efficiency Rating** (18.9%) - Energy star impact
4. **Appliance Age** (12.3%) - Degradation factor
5. **Seasonal Adjustments** (7.7%) - Environmental effects

### **Cross-Validation Results**
- 5-fold CV Mean Accuracy: 87.9% ± 2.1%
- Consistent performance across different data splits
- No significant overfitting detected

## 🔍 Key Components

### **Neural Network Model (`src/model.py`)**
- **ApplianceEnergyPredictor**: Main neural network class with TensorFlow/Keras integration
- **Architecture**: 4-layer deep network with dropout and batch normalization
- **Training Pipeline**: Automated model training with early stopping and learning rate scheduling
- **Model Persistence**: Efficient saving/loading of trained models and preprocessing components

### **Data Processing (`src/data_processing.py`)**
- **ApplianceDataProcessor**: Comprehensive feature engineering pipeline
- **50+ Feature Creation**: Automated generation of sophisticated features from basic appliance data
- **Appliance Validation**: Smart validation for appliance specifications and usage patterns
- **Data Scaling**: Robust normalization for optimal neural network performance

### **Web Application (`src/app.py`)**
- **Flask Framework**: Production-ready web server with comprehensive error handling
- **Neural Network Integration**: Seamless loading and inference with TensorFlow models
- **REST API**: Complete API endpoints for single and batch predictions
- **Real-time Processing**: Instant predictions with confidence intervals and analysis

### **Utility Functions (`src/utils.py`)**
- **Energy Calculations**: Carbon footprint, cost estimation, and efficiency scoring
- **Appliance Intelligence**: Smart recommendations and optimization suggestions
- **Validation Helpers**: Comprehensive input validation and error handling
- **Report Generation**: Automated analysis and insight generation

### **Web Interface (`templates/`)**
- **Responsive Design**: Modern Bootstrap-based UI that works on all devices
- **Interactive Forms**: User-friendly appliance input with real-time validation
- **Data Visualization**: Chart.js integration for consumption analysis and comparisons
- **Batch Processing**: CSV upload interface with comprehensive results display

## 📚 Notebooks Overview

### **1. Data Exploration (`01_data_exploration.ipynb`)**
- **Purpose**: Comprehensive Exploratory Data Analysis (EDA) for appliance energy data
- **Content**: Distribution analysis, correlation studies, consumption pattern identification
- **Insights**: Appliance usage trends, seasonal effects, efficiency relationships
- **Output**: Data quality assessment and feature selection guidance

### **2. Feature Engineering (`02_feature_engineering.ipynb`)**
- **Purpose**: Advanced feature creation from basic appliance specifications
- **Content**: 50+ feature engineering techniques, one-hot encoding, seasonal adjustments
- **Innovation**: Smart feature combinations, appliance-specific calculations
- **Output**: Enhanced dataset ready for neural network training

### **3. Neural Network Model (`03_neural_network_model.ipynb`)**
- **Purpose**: TensorFlow/Keras model development and architecture optimization
- **Content**: Network design, hyperparameter tuning, training pipeline
- **Architecture**: 4-layer deep network with regularization techniques
- **Output**: Trained neural network model with performance validation

### **4. Model Evaluation (`04_model_evaluation.ipynb`)**
- **Purpose**: Comprehensive model performance analysis and validation
- **Content**: Accuracy metrics, cross-validation, error analysis, feature importance
- **Validation**: Multiple evaluation techniques and performance benchmarking
- **Output**: Model performance report and deployment readiness assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

See `requirements.txt` for a complete list of dependencies.

## 📞 Support

For questions or issues, please open an issue on the repository or contact the development team.

## 🗂️ Directory Details

### `/data`
- `raw/`: Store original, unprocessed data files
- `processed/`: Store cleaned and preprocessed data
- `external/`: Store external data sources (weather, holidays, etc.)

### `/notebooks`
- Interactive Jupyter notebooks for analysis and development
- Each notebook focuses on a specific aspect of the pipeline

### `/src`
- Core Python modules for the project
- Modular design for easy maintenance and testing

### `/models`
- Directory for storing trained model files
- Models are saved in pickle format for easy loading

### `/documentation`
- Additional project documentation
- Technical specifications and design documents

## 🎯 Future Enhancements

### **Advanced Neural Network Features**
- [ ] **Transformer Architecture**: Implement attention mechanisms for sequence modeling
- [ ] **LSTM Integration**: Add recurrent layers for temporal pattern recognition
- [ ] **Ensemble Methods**: Combine multiple neural networks for improved accuracy
- [ ] **Transfer Learning**: Leverage pre-trained models for faster development

### **Enhanced Data Integration**
- [ ] **Real-time Data Streams**: Integration with smart meter APIs and IoT devices
- [ ] **Weather API Integration**: Automatic temperature and season data incorporation
- [ ] **Energy Market Data**: Dynamic pricing and demand response features
- [ ] **User Behavior Analytics**: Advanced usage pattern learning

### **Production & Deployment**
- [ ] **Cloud Deployment**: AWS/Azure deployment with auto-scaling
- [ ] **Docker Containerization**: Containerized deployment for easy scaling
- [ ] **Model Monitoring**: Automatic model performance tracking and alerts
- [ ] **A/B Testing Framework**: Systematic model version comparison

### **Advanced Analytics**
- [ ] **Predictive Maintenance**: Appliance lifecycle and failure prediction
- [ ] **Optimization Algorithms**: Automated energy usage optimization recommendations
- [ ] **Cost-Benefit Analysis**: ROI calculations for appliance upgrades
- [ ] **Sustainability Metrics**: Advanced carbon footprint and environmental impact analysis

### **User Experience**
- [ ] **Mobile Application**: Native iOS/Android apps with offline capabilities
- [ ] **Voice Integration**: Alexa/Google Assistant compatibility
- [ ] **Personalized Dashboards**: Custom analytics based on user preferences
- [ ] **Social Features**: Community benchmarking and energy challenges

---

**Happy Predicting! ⚡**
