# ai_models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

def train_all_models(X, y):
    """
    Trains models and returns TEST DATA for validation graphs.
    """
    # 1. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Split (We keep X_test and y_test hidden from the model to validate later)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models_to_train = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    results = {}
    
    for name, model in models_to_train.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        # Save Model + DATA for Validation Graphs
        results[name] = {
            "model_object": model,
            "R2": r2,
            "MAE": mae,
            "validation_data": {
                "Actual": y_test,
                "Predicted": preds
            }
        }
        
    return results, scaler