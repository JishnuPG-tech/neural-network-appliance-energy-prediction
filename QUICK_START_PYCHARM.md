# 🚀 Quick Start Guide for PyCharm

## 1. Install Dependencies

Open PyCharm's terminal and run:

```bash
pip install tensorflow pandas numpy matplotlib seaborn plotly scikit-learn scipy joblib
```

## 2. Run the Scripts

### Option A: Right-click Method
1. Open any script file (e.g., `scripts/01_data_exploration.py`)
2. Right-click in the editor
3. Select "Run '01_data_exploration'"

### Option B: Run Configuration
1. Go to Run → Edit Configurations
2. Click + → Python Script
3. Set Script path to your script file
4. Click OK and Run

### Option C: Terminal Method
```bash
cd "c:\Users\JISHNU PG\Videos\Energy Project\electricity-prediction-project"
python scripts/01_data_exploration.py
python scripts/02_data_preprocessing.py
```

## 3. For Jupyter Notebooks

### PyCharm Professional:
- File → Open → Select .ipynb file
- Run cells with Shift+Enter

### PyCharm Community:
```bash
pip install jupyter
jupyter notebook
```
Then open notebooks in browser.

## 4. Troubleshooting

### If you get import errors:
1. Check Python interpreter: File → Settings → Project → Python Interpreter
2. Install missing packages: `pip install package_name`
3. Make sure you're in the right directory

### If packages won't install:
```bash
pip install --upgrade pip
pip install --no-cache-dir package_name
```

## 🎯 Execution Order

Run these in sequence:
1. `01_data_exploration.py` - Analyze your data
2. `02_data_preprocessing.py` - Clean and prepare data
3. `03_neural_network_model.py` - Train the model
4. `04_model_evaluation.py` - Evaluate results

Each script is standalone and will work in PyCharm!