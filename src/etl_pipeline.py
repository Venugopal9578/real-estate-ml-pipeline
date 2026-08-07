import pandas as pd
import os
from datetime import datetime
from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from data_validation import DataValidator

class ETLPipeline:
    """
    Main ETL orchestration class
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.log = []
        self.start_time = datetime.now()
    
    def log_message(self, message):
        """Log pipeline messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)
        print(log_entry)
    
    def run(self, input_path, output_path):
        """Run complete ETL pipeline"""
        try:
            self.log_message("🚀 Starting ETL Pipeline...")
            
            # EXTRACT
            self.log_message("📥 EXTRACT: Loading raw data...")
            self.log_message(f"   Source: {input_path}")
            
            # TRANSFORM
            self.log_message("🔄 TRANSFORM: Data Cleaning...")
            cleaner = DataCleaner(input_path)
            cleaner.remove_duplicates() \
                   .handle_missing_values() \
                   .data_type_validation() \
                   .remove_outliers(['price_lakhs', 'area_sqft']) \
                   .validate_ranges()
            
            df_cleaned = cleaner.df
            self.log_message(f"   ✓ Cleaned data: {len(df_cleaned)} rows")
            
            # Feature Engineering
            self.log_message("🔄 TRANSFORM: Feature Engineering...")
            engineer = FeatureEngineer(df_cleaned)
            df_engineered = engineer.create_price_per_sqft() \
                                   .create_rooms_ratio() \
                                   .create_amenities_density() \
                                   .create_property_age_category() \
                                   .create_price_category() \
                                   .create_size_category() \
                                   .get_dataframe()
            
            self.log_message(f"   ✓ Features created: {len(engineer.new_features)} new features")
            
            # Validation
            self.log_message("✅ VALIDATE: Data Quality Checks...")
            validator = DataValidator(df_engineered)
            validator.check_nulls().check_duplicates()
            self.log_message("   ✓ Validation passed")
            
            # LOAD
            self.log_message("📤 LOAD: Saving processed data...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_engineered.to_csv(output_path, index=False)
            self.log_message(f"   ✓ Output saved: {output_path}")
            
            # Pipeline complete
            duration = (datetime.now() - self.start_time).total_seconds()
            self.log_message(f"✅ Pipeline Complete! ({duration:.2f}s)")
            
            return {
                'success': True,
                'rows_processed': len(df_engineered),
                'columns': len(df_engineered.columns),
                'duration_seconds': duration,
                'output_path': output_path
            }
        
        except Exception as e:
            self.log_message(f"❌ Pipeline Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def save_log(self, log_path):
        """Save pipeline log"""
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log))
        self.log_message(f"Log saved to {log_path}")


# Main execution
if __name__ == "__main__":
    pipeline = ETLPipeline()
    
    result = pipeline.run(
        input_path='data/raw_data/house_prices_raw.csv',
        output_path='data/processed_data/house_prices_final.csv'
    )
    
    # Save logs
    os.makedirs('logs', exist_ok=True)
    pipeline.save_log(f'logs/etl_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Print result
    if result['success']:
        print(f"\n✅ SUCCESS: Processed {result['rows_processed']} rows in {result['duration_seconds']:.2f}s")