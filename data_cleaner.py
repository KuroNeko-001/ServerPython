import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load data from CSV or Excel file
    """
    print(f"Loading data from {file_path}...")
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    
    # Load data based on file type
    if ext.lower() == '.csv':
        data = pd.read_csv(file_path)
    elif ext.lower() == '.xlsx' or ext.lower() == '.xls':
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV or Excel files are supported.")
    
    print(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
    return data

def parse_content_string(content_str):
    """
    Parse sensor values from a text string in format:
    "INDOOR AIR:1;Temperature:25.2Celsius;humid:60.6%RH;PM2.5:3PPM;CO2:710PPM"
    """
    result = {}
    
    # Split the text into parts separated by ;
    parts = content_str.split(';')
    
    # Process each part
    for part in parts:
        # Skip INDOOR AIR:1
        if 'INDOOR AIR' in part:
            result['location'] = part.split(':')[1] if ':' in part else ''
            continue
            
        # Split name and value by :
        if ':' in part:
            key, value = part.split(':', 1)
            
            # Clean variable name
            key = key.strip().lower()
            
            # Extract only numeric values
            # Use regular expression to extract only numbers (including decimal points)
            numeric_value = re.search(r'[-+]?\d*\.\d+|\d+', value)
            if numeric_value:
                numeric_value = float(numeric_value.group())
                
                # Store value
                if 'temp' in key:
                    result['temperature'] = numeric_value
                elif 'humid' in key:
                    result['humidity'] = numeric_value
                elif 'pm2.5' in key or 'pm2_5' in key:
                    result['pm2_5'] = numeric_value
                elif 'co2' in key:
                    result['co2'] = numeric_value
    
    return result

def process_sensor_data(data, content_column='Content', timestamp_column='Report Time'):
    """
    Convert sensor data from Content column into separate columns
    """
    print("Processing sensor data from content strings...")
    
    # Check if the required column exists in the data
    if content_column not in data.columns:
        raise ValueError(f"Column '{content_column}' not found in the data")
    
    # Create a new DataFrame to store results
    processed_data = pd.DataFrame()
    
    # If there's a timestamp column, keep it
    if timestamp_column in data.columns:
        processed_data[timestamp_column] = data[timestamp_column]
    
    # Convert text strings to separate columns
    parsed_data = data[content_column].apply(parse_content_string)
    
    # Convert list of dicts to DataFrame
    sensor_df = pd.DataFrame([x for x in parsed_data])
    
    # Combine DataFrames
    if len(processed_data) == 0:
        processed_data = sensor_df
    else:
        processed_data = pd.concat([processed_data, sensor_df], axis=1)
    
    print(f"Processed data: {len(processed_data)} rows with sensor values extracted")
    return processed_data

def estimate_missing_sensors(result):
    """
    Estimate missing sensor values (PM10, CO, O3, NO2, SO2)
    using more complex relationships and adding variability to match real-world conditions
    """
    print("Estimating missing sensor values with enhanced variability...")
    
    # Record number of rows before estimation
    initial_rows = len(result)
    
    if 'pm2_5' in result.columns:
        # --- PM10 - Use more complex relationship with PM2.5 ---
        # PM10/PM2.5 ratio varies with PM2.5 level and weather conditions
        # Research shows ratios typically between 1.5-2.2 and varies with humidity
        
        # Function to find PM10/PM2.5 ratio that varies with PM2.5 value and humidity
        def get_pm_ratio(pm25, humidity):
            # Start with base ratio of 1.8
            base_ratio = 1.8
            
            # Adjust based on PM2.5 level (higher PM2.5 usually has lower ratio)
            if pm25 < 20:
                pm_factor = 0.2
            elif pm25 < 50:
                pm_factor = 0.1
            elif pm25 < 100:
                pm_factor = 0
            else:
                pm_factor = -0.1
                
            # Adjust based on humidity (higher humidity usually has lower ratio)
            if humidity < 40:
                humidity_factor = 0.2
            elif humidity < 60:
                humidity_factor = 0.1
            else:
                humidity_factor = -0.1
                
            # Calculate final ratio
            ratio = base_ratio + pm_factor + humidity_factor
            
            # Add slight variability (±0.2)
            import numpy as np
            ratio += np.random.uniform(-0.2, 0.2)
            
            # Set boundaries between 1.5-2.2
            return max(1.5, min(2.2, ratio))
        
        # Create pm10 column from pm2_5 using ratio that varies with pm2_5 and humidity
        result['pm10'] = result.apply(
            lambda row: row['pm2_5'] * get_pm_ratio(row['pm2_5'], row['humidity']), 
            axis=1
        )
        
        print(f"  - Estimated PM10 with variable ratio to PM2.5 based on PM2.5 level and humidity")
    else:
        print("  ! Cannot estimate PM10: PM2.5 data not available")

    # --- CO (Carbon Monoxide) ---
    # CO has relationship with CO2, temperature, and ventilation
    
    if 'co2' in result.columns:
        import numpy as np
        
        # Function to estimate CO from CO2, temperature and humidity
        def estimate_co(co2, temp, humidity):
            # Base value based on CO2 level
            if co2 < 800:
                base_co = 0.7
            elif co2 < 1000:
                base_co = 1.0
            elif co2 < 1500:
                base_co = 1.5
            else:
                base_co = 2.0
            
            # Factor from temperature (higher temperature impacts CO accumulation)
            temp_factor = max(0, (temp - 25) / 50)
            
            # Factor from humidity (lower humidity increases CO concentration)
            humidity_factor = max(0, (60 - humidity) / 150)
            
            # Combine all factors
            co_value = base_co * (1 + temp_factor + humidity_factor)
            
            # Add slight variability (±20%)
            co_value *= np.random.uniform(0.8, 1.2)
            
            # Set appropriate boundaries (0.5-3.0 ppm)
            return max(0.5, min(3.0, co_value))
        
        # Estimate CO values
        result['co'] = result.apply(
            lambda row: estimate_co(
                row['co2'], 
                row['temperature'] if 'temperature' in result.columns else 25, 
                row['humidity'] if 'humidity' in result.columns else 50
            ),
            axis=1
        )
        
        print(f"  - Estimated CO based on CO2, temperature, and humidity with natural variability")
    else:
        print("  ! Cannot estimate CO: CO2 data not available")

    # --- O3 (Ozone) ---
    # O3 has relationship with temperature, humidity, time of day, and UV radiation
    # We'll simulate daily patterns and weather influence
    
    if 'timestamp' in result.columns and 'temperature' in result.columns and 'humidity' in result.columns:
        import numpy as np
        
        # Convert timestamp column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])
        
        # Function to estimate O3 from time, temperature, and humidity
        def estimate_o3(timestamp, temp, humidity):
            # Extract hour of day (0-23)
            hour = timestamp.hour
            
            # Model variation by time of day (O3 higher in afternoon)
            time_factor = np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0
            time_factor = max(0, time_factor)
            
            # Base O3 value
            base_o3 = 0.015
            
            # Factor from temperature (O3 increases with higher temperature)
            temp_factor = max(0, (temp - 25) / 50)
            
            # Factor from humidity (O3 decreases with higher humidity)
            humidity_factor = max(0, (80 - humidity) / 200)
            
            # Combine all factors
            o3_value = base_o3 * (1 + time_factor + temp_factor - humidity_factor)
            
            # Add slight variability (±30%)
            o3_value *= np.random.uniform(0.7, 1.3)
            
            # Set appropriate boundaries (0.01-0.05 ppm)
            return max(0.01, min(0.05, o3_value))
        
        # Estimate O3 values
        result['o3'] = result.apply(
            lambda row: estimate_o3(row['timestamp'], row['temperature'], row['humidity']),
            axis=1
        )
        
        print(f"  - Estimated O3 based on time of day, temperature, and humidity patterns")
    else:
        # If no time data or temperature/humidity, use random values
        import numpy as np
        result['o3'] = np.random.uniform(0.01, 0.03, size=len(result))
        print(f"  - Estimated O3 with random values between 0.01-0.03 ppm (limited data available)")

    # --- NO2 (Nitrogen Dioxide) ---
    # NO2 has relationship with CO2, ventilation, and combustion
    
    if 'co2' in result.columns:
        import numpy as np
        
        # Function to estimate NO2 from CO2 and CO
        def estimate_no2(co2, co, temp):
            # Base value based on CO2 level
            if co2 < 800:
                base_no2 = 0.02
            elif co2 < 1000:
                base_no2 = 0.03
            elif co2 < 1500:
                base_no2 = 0.04
            else:
                base_no2 = 0.05
            
            # Factor from CO (indicating combustion)
            co_factor = co / 10
            
            # Factor from temperature (higher temperature increases NO2)
            temp_factor = max(0, (temp - 20) / 100)
            
            # Combine all factors
            no2_value = base_no2 + co_factor + temp_factor
            
            # Add slight variability (±25%)
            no2_value *= np.random.uniform(0.75, 1.25)
            
            # Set appropriate boundaries (0.01-0.1 ppm)
            return max(0.01, min(0.1, no2_value))
        
        # Estimate NO2 values
        result['no2'] = result.apply(
            lambda row: estimate_no2(
                row['co2'], 
                row['co'], 
                row['temperature'] if 'temperature' in result.columns else 25
            ),
            axis=1
        )
        
        print(f"  - Estimated NO2 based on CO2, CO, and temperature with natural variability")
    else:
        # If no CO2 data, use random values
        import numpy as np
        result['no2'] = np.random.uniform(0.02, 0.06, size=len(result))
        print(f"  - Estimated NO2 with random values between 0.02-0.06 ppm (limited data available)")

    # --- SO2 (Sulfur Dioxide) ---
    # SO2 in buildings is usually low, but still has variability and may have relationship with
    # temperature, CO2, and external factors (e.g., outdoor pollution)
    
    if 'co2' in result.columns and 'pm2_5' in result.columns:
        import numpy as np
        
        # Function to estimate SO2 from PM2.5 (indicating outdoor pollution) and CO2
        def estimate_so2(pm25, co2):
            # Base low value for general buildings
            base_so2 = 0.003
            
            # Factor from PM2.5 (indicating outdoor pollution that may have SO2)
            pm_factor = pm25 / 1000
            
            # Factor from CO2 (indicating ventilation)
            co2_factor = max(0, (co2 - 800) / 10000)
            
            # Combine all factors
            so2_value = base_so2 + pm_factor + co2_factor
            
            # Add realistic variability (±40%)
            # SO2 typically has high variability even at low levels
            so2_value *= np.random.uniform(0.6, 1.4)
            
            # 5% chance of having high erroneous value (simulating outdoor pollution)
            if np.random.random() < 0.05:
                so2_value *= np.random.uniform(2, 3)
            
            # Set appropriate boundaries (0.001-0.02 ppm)
            return max(0.001, min(0.02, so2_value))
        
        # Estimate SO2 values
        result['so2'] = result.apply(
            lambda row: estimate_so2(row['pm2_5'], row['co2']),
            axis=1
        )
        
        print(f"  - Estimated SO2 based on PM2.5 and CO2 with realistic variability")
    else:
        # If no PM2.5 or CO2 data, use random values
        import numpy as np
        result['so2'] = np.random.uniform(0.001, 0.01, size=len(result))
        print(f"  - Estimated SO2 with random values between 0.001-0.01 ppm (limited data available)")
    
    # Check if estimation was successful
    print(f"  • Estimated sensor values for {len(result)} rows")
    print(f"  • Before: {initial_rows} rows, After: {len(result)} rows")
    
    # Show statistics of estimated values
    print("\nStatistics of estimated values:")
    estimated_cols = ['pm10', 'co', 'o3', 'no2', 'so2']
    estimated_stats = result[estimated_cols].describe().round(4)
    print(estimated_stats)
    
    return result

def clean_data(data):
    """
    Clean and preprocess data by handling missing values and outliers
    """
    print("Cleaning and validating data...")
    
    # Copy data to avoid changing original data
    cleaned_data = data.copy()
    
    # Check for missing values
    missing_values = cleaned_data.isnull().sum()
    print(f"Missing values before cleaning:\n{missing_values}")
    
    # Fill missing values with mean (or other appropriate method)
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if cleaned_data[col].isnull().sum() > 0:
            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
            print(f"- Filled missing values in {col} with mean value")
    
    # Check for outliers using IQR method
    for col in numeric_columns:
        Q1 = cleaned_data[col].quantile(0.25)
        Q3 = cleaned_data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outlier values
        outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"- Found {outliers} outliers in {col}")
            
            # Adjust outlier values by capping upper/lower bounds
            cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
            cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
            print(f"  - Capped outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Add timestamp column if exists
    if 'Report Time' in cleaned_data.columns:
        try:
            cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['Report Time'])
            print("- Converted 'Report Time' to datetime format")
        except:
            print("- Could not convert 'Report Time' to datetime format")
    
    print(f"Data cleaning complete: {len(cleaned_data)} rows retained")
    return cleaned_data

def create_aqi_labels(data):
    """
    Create Air Quality Index (AQI) class labels based on PM2.5 values
    according to Thailand AQI standards
    """
    print("Creating AQI class labels according to Thai standards...")
    
    # Copy data
    result = data.copy()
    
    # Check if pm2_5 column exists
    if 'pm2_5' not in result.columns:
        raise ValueError("Column 'pm2_5' not found in the data, cannot create AQI class labels")
    
    # Convert PM2.5 values from PPM to μg/m³ if necessary (if values are in PPM, they're low)
    # Normal PM2.5 values should be between 0-1000 μg/m³
    if result['pm2_5'].max() < 1.0:
        print("- PM2.5 values appear to be in PPM, converting to μg/m³")
        result['pm2_5'] = result['pm2_5'] * 1000
    
    # Function to assign AQI class based on PM2.5 value (Thailand standards)
    def assign_aqi_class(pm25):
        if pm25 <= 25:
            return 0  # Excellent (0-25)
        elif pm25 <= 37:
            return 1  # Good (26-37)
        elif pm25 <= 50:
            return 2  # Moderate (38-50)
        elif pm25 <= 90:
            return 3  # Starting to affect health (51-90)
        else:
            return 4  # Affecting health (91+)
    
    # Create AQI Class column
    result['aqi_class'] = result['pm2_5'].apply(assign_aqi_class)
    
    # Show distribution of classes
    class_counts = result['aqi_class'].value_counts().sort_index()
    print("\nNumber of samples in each AQI level:")
    aqi_labels = {
        0: "Excellent (0-25 μg/m³)",
        1: "Good (26-37 μg/m³)",
        2: "Moderate (38-50 μg/m³)",
        3: "Starting to affect health (51-90 μg/m³)",
        4: "Affecting health (91+ μg/m³)"
    }
    
    for aqi_class, count in class_counts.items():
        print(f"  Level {aqi_class} - {aqi_labels[aqi_class]}: {count} samples ({count/len(result)*100:.1f}%)")
    
    return result

def visualize_data(data):
    """
    Create visualizations to understand the dataset
    """
    print("\nCreating visualizations of cleaned data...")
    
    # Create directory for saving images if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Pairplot for key features
    key_features = ['temperature', 'humidity', 'pm2_5', 'co2']
    if 'aqi_class' in data.columns:
        try:
            plt.figure(figsize=(12, 10))
            sns.pairplot(data[key_features + ['aqi_class']], hue='aqi_class', palette='viridis')
            plt.suptitle('Relationships Between Key Features', y=1.02)
            plt.savefig('visualizations/key_features_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("- Created pairplot of key features")
        except:
            print("- Could not create pairplot (possible missing data)")
    
    # 2. Correlation heatmap
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 10))
        correlation = numeric_data.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
        plt.title('Correlation Between Features')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
        plt.close()
        print("- Created correlation heatmap")
    except:
        print("- Could not create correlation heatmap")
    
    # 3. Time series plot if timestamp exists
    if 'timestamp' in data.columns:
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(data['timestamp'], data['pm2_5'], 'b-', label='PM2.5')
            plt.title('PM2.5 Over Time')
            plt.ylabel('PM2.5 (μg/m³)')
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(data['timestamp'], data['temperature'], 'r-', label='Temperature')
            plt.title('Temperature Over Time')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(data['timestamp'], data['co2'], 'g-', label='CO2')
            plt.title('CO2 Over Time')
            plt.ylabel('CO2 (PPM)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/time_series.png', dpi=300)
            plt.close()
            print("- Created time series plots")
        except:
            print("- Could not create time series plots")
    
    # 4. Distribution of estimated vs original values (if both exist)
    if 'pm2_5' in data.columns and 'pm10' in data.columns:
        try:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(data['pm2_5'], kde=True, color='blue')
            plt.title('PM2.5 Distribution (Original)')
            
            plt.subplot(1, 2, 2)
            sns.histplot(data['pm10'], kde=True, color='green')
            plt.title('PM10 Distribution (Estimated)')
            
            plt.tight_layout()
            plt.savefig('visualizations/estimated_vs_original.png', dpi=300)
            plt.close()
            print("- Created distributions of original vs estimated values")
        except:
            print("- Could not create distribution plots")
    
    print("Visualizations saved to 'visualizations' folder")

def save_cleaned_data(data, output_file='cleaned_data.csv'):
    """
    Save cleaned and processed data to CSV file
    """
    print(f"\nSaving cleaned data to {output_file}...")
    data.to_csv(output_file, index=False)
    print(f"Data saved successfully with {len(data)} rows and {len(data.columns)} columns")
    
    # Show sample of saved data
    print("\nSample of cleaned data:")
    print(data.head())
    
    # Show details of each column
    print("\nColumn details:")
    for col in data.columns:
        if data[col].dtype == 'object':
            print(f"- {col}: {data[col].dtype} (examples: {', '.join(str(x) for x in data[col].head(3).values)})")
        else:
            print(f"- {col}: {data[col].dtype} (min: {data[col].min():.2f}, max: {data[col].max():.2f}, mean: {data[col].mean():.2f})")
    
    return output_file

def main():
    """
    Main function to execute the data cleaning process
    """
    print("=" * 80)
    print("Indoor Air Quality Data Cleaning and Preprocessing")
    print("=" * 80)
    
    # Use data file directly from specified location
    file_path = '/Users/madhood/Documents/CPE495/Project-Simmulate/Data-non-cleansing.xlsx'
    print(f"\nLoading data from file: {file_path}")
    
    try:
        # Load data
        data = load_data(file_path)
        
        # Show sample of raw data
        print("\nSample of raw data:")
        print(data.head())
        
        # Check if Content column exists
        if 'Content' in data.columns:
            # Convert text from Content column to separate columns
            data = process_sensor_data(data, content_column='Content', timestamp_column='Report Time')
        else:
            print("No 'Content' column found, assuming data is already in structured format")
        
        # Clean data
        cleaned_data = clean_data(data)
        
        # Estimate missing sensor values
        enriched_data = estimate_missing_sensors(cleaned_data)
        
        # Create AQI Class
        labeled_data = create_aqi_labels(enriched_data)
        
        # Create visualization
        visualize_data(labeled_data)
        
        # Save data
        output_file = save_cleaned_data(labeled_data)
        
        print("\nData cleaning and preprocessing complete!")
        print(f"You can now use the cleaned data file '{output_file}' with train_model_rf.py")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Process terminated due to error.")
        
if __name__ == "__main__":
    main() 