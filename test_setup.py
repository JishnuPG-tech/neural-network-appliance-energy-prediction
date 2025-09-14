"""
Test script to verify the electricity prediction project setup
"""

def test_imports():
    """Test if all core libraries can be imported"""
    print("Testing core library imports...")
    
    try:
        import pandas as pd
        print("✅ pandas:", pd.__version__)
    except ImportError:
        print("❌ pandas not available")
        return False
    
    try:
        import numpy as np
        print("✅ numpy:", np.__version__)
    except ImportError:
        print("❌ numpy not available")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn:", sklearn.__version__)
    except ImportError:
        print("❌ scikit-learn not available")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib:", matplotlib.__version__)
    except ImportError:
        print("❌ matplotlib not available")
        return False
    
    try:
        import seaborn as sns
        print("✅ seaborn:", sns.__version__)
    except ImportError:
        print("❌ seaborn not available")
        return False
    
    try:
        import flask
        print("✅ Flask:", flask.__version__)
    except ImportError:
        print("❌ Flask not available")
        return False
    
    try:
        import plotly
        print("✅ plotly:", plotly.__version__)
    except ImportError:
        print("❌ plotly not available")
        return False
    
    try:
        import joblib
        print("✅ joblib:", joblib.__version__)
    except ImportError:
        print("❌ joblib not available")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Create sample data
        X = np.random.randn(100, 3)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        df['target'] = y
        
        # Train simple model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make prediction
        prediction = model.predict(X[:1])
        
        print("✅ Basic ML pipeline working")
        print(f"   Sample prediction: {prediction[0]:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ELECTRICITY PREDICTION PROJECT - SETUP TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 Setup verification successful!")
            print("Your environment is ready for the electricity prediction project.")
            print("\nNext steps:")
            print("1. Add your data to the data/raw/ folder")
            print("2. Open notebooks/01_data_exploration.ipynb")
            print("3. Follow the notebook sequence for your analysis")
        else:
            print("\n⚠️  Basic functionality test failed")
    else:
        print("\n❌ Some required libraries are missing")
        print("Please install missing packages using pip install")
    
    print("=" * 50)
