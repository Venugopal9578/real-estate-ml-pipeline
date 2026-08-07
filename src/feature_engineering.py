import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Create new features from existing data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.new_features = []
    
    def create_price_per_sqft(self):
        """Price per square foot"""
        self.df['price_per_sqft'] = (self.df['price_lakhs'] * 100000) / self.df['area_sqft']
        self.new_features.append('price_per_sqft')
        return self
    
    def create_rooms_ratio(self):
        """Ratio of bedrooms to bathrooms"""
        self.df['rooms_ratio'] = self.df['bedrooms'] / (self.df['bathrooms'] + 1)
        self.new_features.append('rooms_ratio')
        return self
    
    def create_amenities_density(self):
        """Amenities per 1000 sqft"""
        self.df['amenities_density'] = (self.df['amenities_count'] / self.df['area_sqft']) * 1000
        self.new_features.append('amenities_density')
        return self
    
    def create_property_age_category(self):
        """Categorize property age"""
        self.df['age_category'] = pd.cut(
            self.df['age_years'],
            bins=[0, 2, 5, 10, 50],
            labels=['Brand New', 'Recent', 'Medium', 'Old']
        )
        self.new_features.append('age_category')
        return self
    
    def create_price_category(self):
        """Categorize price range"""
        self.df['price_category'] = pd.cut(
            self.df['price_lakhs'],
            bins=[0, 100, 200, 300, 500],
            labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
        )
        self.new_features.append('price_category')
        return self
    
    def create_size_category(self):
        """Categorize property size"""
        self.df['size_category'] = pd.cut(
            self.df['area_sqft'],
            bins=[0, 1000, 2000, 3000, 5000],
            labels=['Compact', 'Standard', 'Large', 'Spacious']
        )
        self.new_features.append('size_category')
        return self
    
    def get_dataframe(self):
        """Return dataframe with new features"""
        return self.df
    
    def get_feature_summary(self):
        """Summary of new features created"""
        return {
            'new_features': self.new_features,
            'total_features': len(self.df.columns)
        }


# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed_data/house_prices_cleaned.csv')
    
    engineer = FeatureEngineer(df)
    df_engineered = engineer.create_price_per_sqft() \
                           .create_rooms_ratio() \
                           .create_amenities_density() \
                           .create_property_age_category() \
                           .create_price_category() \
                           .create_size_category() \
                           .get_dataframe()
    
    # Save
    df_engineered.to_csv('data/processed_data/house_prices_featured.csv', index=False)
    
    # Print summary
    summary = engineer.get_feature_summary()
    print(f"\n✓ Created {len(summary['new_features'])} new features")
    print(f"✓ Total features now: {summary['total_features']}")