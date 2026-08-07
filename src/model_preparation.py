import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class ModelDataPreparation:
    """
    Prepare data for machine learning models
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def separate_features_target(self, target_col='price_lakhs'):
        """Separate features and target variable"""
        self.y = self.df[target_col]
        
        # Drop non-feature columns
        drop_cols = [target_col, 'property_id', 'age_category', 'price_category', 'size_category']
        self.X = self.df.drop(columns=drop_cols)
        
        print(f"✓ Features shape: {self.X.shape}")
        print(f"✓ Target shape: {self.y.shape}")
        return self
    
    def encode_categorical(self):
        """Encode categorical features"""
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            self.label_encoders[col] = le
            print(f"✓ Encoded {col}: {len(le.classes_)} unique values")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        return self
    
    def scale_features(self):
        """Scale numerical features"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for compatibility
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.X_test.columns
        )
        
        print(f"✓ Features scaled (0 mean, 1 std)")
        return self
    
    def get_prepared_data(self):
        """Return prepared data"""
        return {
            'X_train': self.X_train_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.X.columns.tolist()
        }
    
    def save_scaler(self, path):
        """Save scaler for future use"""
        pickle.dump(self.scaler, open(path, 'wb'))
        print(f"✓ Scaler saved to {path}")


# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed_data/house_prices_final.csv')
    
    prep = ModelDataPreparation(df)
    prep.separate_features_target() \
        .encode_categorical() \
        .split_data() \
        .scale_features()
    
    # Save scaler
    prep.save_scaler('models/scaler.pkl')
    
    # Get prepared data
    data = prep.get_prepared_data()
    print(f"\n✅ Data preparation complete!")
    print(f"   Features: {len(data['feature_names'])}")