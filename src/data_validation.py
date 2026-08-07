import pandas as pd

class DataValidator:
    """
    Validate data quality and schema
    """
    
    def __init__(self, df):
        self.df = df
        self.validation_results = {}
    
    def check_nulls(self):
        """Check for null values"""
        null_counts = self.df.isnull().sum()
        self.validation_results['nulls'] = null_counts.to_dict()
        if null_counts.sum() == 0:
            print("✓ No null values found")
        return self
    
    def check_duplicates(self):
        """Check for duplicates"""
        dup_count = self.df.duplicated().sum()
        self.validation_results['duplicates'] = dup_count
        print(f"✓ Duplicate rows: {dup_count}")
        return self
    
    def check_schema(self, expected_types):
        """Validate column data types"""
        schema_valid = True
        for col, dtype in expected_types.items():
            if col in self.df.columns:
                if str(self.df[col].dtype) != str(dtype):
                    print(f"⚠ {col}: expected {dtype}, got {self.df[col].dtype}")
                    schema_valid = False
        if schema_valid:
            print("✓ Schema validation passed")
        self.validation_results['schema'] = schema_valid
        return self
    
    def check_value_ranges(self, range_rules):
        """Validate values are within expected ranges"""
        for col, (min_val, max_val) in range_rules.items():
            invalid = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            if len(invalid) > 0:
                print(f"⚠ {col}: {len(invalid)} values out of range [{min_val}, {max_val}]")
            else:
                print(f"✓ {col}: all values within range")
        return self
    
    def get_report(self):
        """Return validation report"""
        return self.validation_results


# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed_data/house_prices_featured.csv')
    
    expected_types = {
        'property_id': 'int64',
        'location': 'object',
        'price_lakhs': 'float64',
        'area_sqft': 'float64'
    }
    
    range_rules = {
        'price_lakhs': (20, 500),
        'area_sqft': (500, 5000),
        'bedrooms': (1, 5)
    }
    
    validator = DataValidator(df)
    validator.check_nulls() \
            .check_duplicates() \
            .check_schema(expected_types) \
            .check_value_ranges(range_rules)
    
    print("\n=== VALIDATION REPORT ===")
    report = validator.get_report()
    print(report)