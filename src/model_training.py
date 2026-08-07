import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import json

class ModelTrainer:
    """
    Train and evaluate multiple machine learning models
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        print("\n📊 Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        print("✓ Linear Regression trained")
        return self
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("✓ Random Forest trained")
        return self
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        print("✓ XGBoost trained")
        return self
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📈 Evaluating Models...")
        
        for model_name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            self.predictions[model_name] = y_pred
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.metrics[model_name] = {
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'R² Score': round(r2, 4),
                'MSE': round(mse, 4)
            }
            
            print(f"\n{model_name}:")
            print(f"  RMSE: ₹{rmse:.2f} lakhs")
            print(f"  MAE:  ₹{mae:.2f} lakhs")
            print(f"  R²:   {r2:.4f}")
        
        return self
    
    def get_best_model(self):
        """Return best performing model"""
        best_model_name = max(
            self.metrics, 
            key=lambda x: self.metrics[x]['R² Score']
        )
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, path):
        """Save all trained models"""
        for model_name, model in self.models.items():
            model_path = f"{path}/{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_path, 'wb'))
            print(f"✓ {model_name} saved")
    
    def get_metrics_report(self):
        """Return metrics as DataFrame"""
        return pd.DataFrame(self.metrics).T


# Usage
if __name__ == "__main__":
    # Load prepared data
    from model_preparation import ModelDataPreparation
    
    df = pd.read_csv('data/processed_data/house_prices_final.csv')
    prep = ModelDataPreparation(df)
    prep.separate_features_target() \
        .encode_categorical() \
        .split_data() \
        .scale_features()
    
    data = prep.get_prepared_data()
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_linear_regression(data['X_train'], data['y_train']) \
           .train_random_forest(data['X_train'], data['y_train']) \
           .train_xgboost(data['X_train'], data['y_train']) \
           .evaluate_models(data['X_test'], data['y_test'])
    
    # Save models
    import os
    os.makedirs('models', exist_ok=True)
    trainer.save_models('models')
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\n🏆 Best Model: {best_name}")
    
    # Save metrics
    metrics_df = trainer.get_metrics_report()
    metrics_df.to_csv('results/model_metrics.csv')
    print("✓ Metrics saved to results/model_metrics.csv")