# 🐍 PyCharm Setup Guide for Neural Network Project

## 📋 Prerequisites Setup

### Step 1: Install Required Packages

First, you need to install all the required Python packages. Open PyCharm's terminal and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install tensorflow==2.13.0
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy
pip install jupyter ipykernel
```

### Step 2: Configure Python Interpreter

1. **Open PyCharm Settings** (File → Settings or Ctrl+Alt+S)
2. **Go to Project → Python Interpreter**
3. **Add New Interpreter** if needed:
   - Click the gear icon → Add
   - Choose "Virtualenv Environment" → New environment
   - Set location: `your_project_folder\.venv`
   - Base interpreter: Python 3.8 or higher

### Step 3: Install Jupyter Support (if using PyCharm Community)

```bash
pip install jupyter notebook
```

## 🚀 Running Methods in PyCharm

### Method 1: Jupyter Notebook Interface (PyCharm Professional)

**Best for:** Interactive development and data exploration

1. **Open Notebook:**
   - File → Open → Select `.ipynb` file
   - PyCharm opens it in notebook interface

2. **Run Cells:**
   - Click ▶️ button next to each cell
   - Use `Shift + Enter` to run and move to next cell
   - Use `Ctrl + Enter` to run current cell

3. **Run All:**
   - Right-click → "Run All Cells"
   - Or use toolbar buttons

### Method 2: Convert to Python Scripts (All PyCharm Versions)

**Best for:** Production code and easier debugging

I've created Python script versions for you:

**To run the data exploration script:**

1. **Open:** `scripts/01_data_exploration.py`
2. **Right-click** → "Run '01_data_exploration'"
3. **Or use:** Ctrl+Shift+F10

### Method 3: Jupyter in External Browser (PyCharm Community)

**Alternative for Community Edition:**

1. **Install Jupyter:**
   ```bash
   pip install jupyter
   ```

2. **Start Jupyter Server:**
   ```bash
   jupyter notebook
   ```

3. **Open in Browser:**
   - Navigate to `notebooks/` folder
   - Open your `.ipynb` files

## 🔧 PyCharm Configuration Tips

### Enable Jupyter Support
1. **Settings** → **Languages & Frameworks** → **Jupyter**
2. **Check:** "Enable Jupyter support"
3. **Set:** Jupyter server URL (usually http://localhost:8888)

### Configure Scientific Mode
1. **Settings** → **Tools** → **Python Scientific**
2. **Enable:** "Show plots in tool window"
3. **Enable:** "Show variables in tool window"

### Optimize for Data Science
1. **Install Plugin:** "Python Scientific" (if not installed)
2. **View** → **Tool Windows** → **SciView**
3. **View** → **Tool Windows** → **Python Packages**

## 📁 Project Structure for PyCharm

```
electricity-prediction-project/
├── scripts/                   # Python scripts (for PyCharm)
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_neural_network_model.py
│   └── 04_model_evaluation.py
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_neural_network_model.ipynb
│   └── 04_model_evaluation.ipynb
├── data/                      # Data files
├── src/                       # Source code modules
├── requirements.txt           # Python dependencies
└── README.md
```

## 🎯 Quick Start Commands for PyCharm

### Run Data Exploration:
```python
# In PyCharm terminal or Python console:
exec(open('scripts/01_data_exploration.py').read())
```

### Interactive Python Console:
1. **Tools** → **Python Console**
2. Run commands interactively:
```python
import pandas as pd
import numpy as np
# ... your code here
```

### Debug Mode:
1. **Set breakpoints** by clicking left margin
2. **Right-click** → "Debug '01_data_exploration'"
3. **Use debug controls** to step through code

## 🔍 Troubleshooting

### Import Errors:
- **Check Python interpreter** is correctly configured
- **Install missing packages:** `pip install package_name`
- **Verify virtual environment** is activated

### Jupyter Issues:
- **Restart Jupyter kernel:** Kernel → Restart
- **Clear outputs:** Cell → All Output → Clear
- **Check port conflicts:** Try different port (8889, 8890)

### Performance:
- **Close unused tool windows**
- **Increase PyCharm memory:** Help → Change Memory Settings
- **Use GPU for TensorFlow:** Runtime → Change runtime type

## 📊 Running Order

**Recommended execution sequence:**

1. **01_data_exploration.py** - Understand your data
2. **02_data_preprocessing.py** - Clean and prepare data  
3. **03_neural_network_model.py** - Build and train model
4. **04_model_evaluation.py** - Evaluate performance

Each script is self-contained and can be run independently in PyCharm!