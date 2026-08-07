import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaner:
    """
    Handle data cleaning and validation
    """
    
    def __init__(self, input_path):
        self.df = pd.read_csv(input_path)
        self.cleaning_log = []
    
    def remove_duplicates(self):
        """Remove duplicate records"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        self.cleaning_log.append(f"Removed {removed} duplicates")
        return self
    
    def handle_missing_values(self):
        """Handle missing values"""
        # For this dataset, there are none, but good practice to have this
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    self.cleaning_log.append(f"Filled {col} with median")
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    self.cleaning_log.append(f"Filled {col} with mode")
        return self
    
    def data_type_validation(self):
        """Ensure correct data types"""
        self.df['property_id'] = self.df['property_id'].astype('int64')
        self.df['bedrooms'] = self.df['bedrooms'].astype('int64')
        self.df['bathrooms'] = self.df['bathrooms'].astype('int64')
        self.df['parking_spaces'] = self.df['parking_spaces'].astype('int64')
        self.df['amenities_count'] = self.df['amenities_count'].astype('int64')
        self.cleaning_log.append("Data types validated")
        return self
    
    def remove_outliers(self, columns, threshold=3):
        """Remove statistical outliers using Z-score"""
        initial_count = len(self.df)
        
        for col in columns:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < threshold]
        
        removed = initial_count - len(self.df)
        self.cleaning_log.append(f"Removed {removed} outliers")
        return self
    
    def validate_ranges(self):
        """Validate data is within expected ranges"""
        validation_rules = {
            'bedrooms': (1, 5),
            'bathrooms': (1, 4),
            'price_lakhs': (20, 500),
            'area_sqft': (500, 5000),
            'parking_spaces': (0, 3)
        }
        
        for col, (min_val, max_val) in validation_rules.items():
            invalid = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            if len(invalid) > 0:
                self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
                self.cleaning_log.append(f"Removed {len(invalid)} invalid {col} values")
        
        return self
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data"""
        self.df.to_csv(output_path, index=False)
        self.cleaning_log.append(f"Saved cleaned data to {output_path}")
        return self
    
    def get_report(self):
        """Return cleaning report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'rows_final': len(self.df),
            'columns': len(self.df.columns),
            'cleaning_steps': self.cleaning_log
        }
        return report


# Usage
if __name__ == "__main__":
    cleaner = DataCleaner('data/raw_data/house_prices_raw.csv')
    
    cleaner.remove_duplicates() \
           .handle_missing_values() \
           .data_type_validation() \
           .remove_outliers(['price_lakhs', 'area_sqft']) \
           .validate_ranges() \
           .save_cleaned_data('data/processed_data/house_prices_cleaned.csv')
    
    # Print report
    report = cleaner.get_report()
    print("\n=== DATA CLEANING REPORT ===")
    for step in report['cleaning_steps']:
        print(f"✓ {step}")
    print(f"\nFinal dataset: {report['rows_final']} rows, {report['columns']} columns")