import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    'property_id': range(1, 5001),
    'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'], 5000),
    'area_sqft': np.random.uniform(500, 5000, 5000),
    'bedrooms': np.random.choice([1, 2, 3, 4, 5], 5000),
    'bathrooms': np.random.choice([1, 2, 3, 4], 5000),
    'age_years': np.random.uniform(0, 50, 5000),
    'price_lakhs': np.random.uniform(20, 500, 5000),
    'parking_spaces': np.random.choice([0, 1, 2, 3], 5000),
    'amenities_count': np.random.randint(0, 10, 5000),
}

df = pd.DataFrame(data)
df.to_csv('data/raw_data/house_prices_raw.csv', index=False)
print("Dataset created!")
print(df.head())