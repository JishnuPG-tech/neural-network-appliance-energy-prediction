# Installation Guide for Electricity Prediction Project

## Quick Setup (Recommended)

### Option 1: Install Core Requirements Only
```bash
pip install -r requirements.txt
```

### Option 2: Use Conda (Recommended for Windows)
```bash
# Create conda environment
conda create -n electricity-pred python=3.9
conda activate electricity-pred

# Install core packages via conda (avoids compilation issues)
conda install pandas numpy scikit-learn matplotlib seaborn jupyter flask plotly
conda install -c conda-forge xgboost lightgbm

# Install remaining packages via pip
pip install tqdm python-dotenv flask-cors loguru flask-restx
```

### Option 3: Install with Pre-compiled Wheels
```bash
# Install numpy first (pre-compiled)
pip install numpy

# Then install other packages
pip install -r requirements.txt
```

## Troubleshooting Windows Installation Issues

### If you encounter compilation errors:

1. **Install Microsoft Visual C++ Build Tools**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ build tools" workload

2. **Use Conda Instead of Pip**
   - Conda provides pre-compiled packages
   - Avoids compilation issues on Windows

3. **Install Packages One by One**
   ```bash
   pip install pandas
   pip install numpy
   pip install scikit-learn
   pip install matplotlib
   pip install seaborn
   pip install jupyter
   pip install flask
   ```

4. **Use Alternative Package Versions**
   ```bash
   # For problematic packages, try:
   pip install --only-binary=all package_name
   ```

## Minimal Setup for Quick Start

If you want to get started quickly with just the essentials:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter flask
```

This will give you enough to run the basic functionality.

## Advanced Packages (Optional)

After core installation works, you can add advanced features:

```bash
pip install -r requirements-optional.txt
```

## Environment Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv electricity-env
   electricity-env\Scripts\activate  # Windows
   ```

2. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

## Verification

Test your installation:
```python
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import flask
print("All core packages installed successfully!")
```

## Common Issues and Solutions

### NumPy Compilation Error
- **Solution**: Use conda or install pre-compiled wheel
- **Command**: `conda install numpy` or `pip install numpy --only-binary=all`

### Missing C++ Compiler
- **Solution**: Install Visual C++ Build Tools
- **Alternative**: Use conda environment

### Package Conflicts
- **Solution**: Create fresh virtual environment
- **Command**: Use conda instead of pip

### Memory Issues During Installation
- **Solution**: Install packages one by one
- **Command**: `pip install --no-cache-dir package_name`

## Success Path (Recommended)

1. Install Anaconda or Miniconda
2. Create conda environment with Python 3.9
3. Install core packages via conda
4. Install remaining packages via pip
5. Verify installation

This approach avoids most Windows compilation issues and provides a stable environment for the electricity prediction project.
