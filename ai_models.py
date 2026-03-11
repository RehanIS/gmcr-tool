import pandas as pd
import numpy as np
import os
import joblib  # 📦 FOR SAVING MODELS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def save_artifacts(model, scaler, platform_name):
    """
    Saves the trained model and scaler to the 'models/' folder.
    """
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save Model
    model_path = f"models/{platform_name}_model.pkl"
    joblib.dump(model, model_path)
    
    # Save Scaler (CRITICAL: We need this to scale new inputs same as training data)
    scaler_path = f"models/{platform_name}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"💾 Artifacts saved: {model_path} & {scaler_path}")

def train_model(X, y, model_choice, platform_name="generic"):
    """
    Trains a model, calculates metrics, and SAVES it to disk.
    """
    # 1. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Initialize Model
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    else:
        # Fallback
        model = GradientBoostingRegressor(random_state=42)
        
    # 4. Train
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    metrics = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds)
    }
    
    # 6. SAVE TO DISK
    save_artifacts(model, scaler, platform_name)
    
    # Return artifacts so main.py can use them immediately
    return model, scaler, metrics