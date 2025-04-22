import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import joblib
import json
import os

# นำเข้าคลาส DataPreprocessor จากไฟล์แยก
from data_preprocessor import DataPreprocessor

# ประกาศตัวแปร preprocessor ในขอบเขตหลัก
preprocessor = DataPreprocessor()

# Load and prepare dataset
def load_dataset(file_path):
    print(f"Loading dataset from {file_path}...")
    
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

# Data cleaning and preprocessing
def preprocess_data(data):
    print("Cleaning and preprocessing data...")
    
    # Show initial data info
    print("\nInitial data preview:")
    print(data.head())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    
    # Drop rows with missing values or fill them
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values, handling them...")
        # Option 1: Drop rows with missing values
        data_cleaned = data.dropna()
        print(f"After removing rows with missing values, {len(data_cleaned)} rows remain")
        
        # Option 2: Fill missing values (uncomment if you prefer this approach)
        # data_cleaned = data.fillna(data.mean())
        # print("Missing values filled with column means")
    else:
        data_cleaned = data
        print("No missing values found")
    
    return data_cleaned

# Create AQI class labels if not already present
def create_aqi_labels(data, pm25_column='pm2_5'):
    print("Creating AQI class labels...")
    
    # Check if 'aqi_class' already exists in the data
    if 'aqi_class' in data.columns:
        print("Found 'aqi_class' column in the data, using existing values")
        return data
    
    # Check if PM2.5 column exists
    if pm25_column not in data.columns:
        raise ValueError(f"Column {pm25_column} not found in the data, cannot create AQI class labels")
    
    # Create AQI class based on PM2.5 values AND temperature AND thermal comfort
    def assign_aqi_class(row):
        # เริ่มจากการกำหนดด้วย PM2.5 ตามเกณฑ์มาตรฐาน
        pm25 = row[pm25_column]
        if pm25 <= 25:
            aqi = 0  # Very Good
        elif pm25 <= 37:
            aqi = 1  # Good
        elif pm25 <= 50:
            aqi = 2  # Moderate
        elif pm25 <= 90:
            aqi = 3  # Unhealthy
        else:
            aqi = 4  # Very Unhealthy
            
        # ปรับแต่งด้วยอุณหภูมิ (ถ้ามี)
        if 'temperature' in row.index:
            temp = row['temperature']
            if temp >= 40:  # อันตราย (ร้อนจัด)
                aqi = max(aqi, 4)  # ต้องเป็นระดับแย่ที่สุด
            elif temp >= 38:  # ไม่ดีต่อสุขภาพ (ร้อนมาก)
                aqi = max(aqi, 3)
            elif temp >= 35:  # มีผลต่อคนอ่อนไหว (ร้อน)
                aqi = max(aqi, 2)
            elif temp <= 10:  # หนาวมาก
                aqi = max(aqi, 3)
            elif temp <= 15:  # หนาวเกินไป
                aqi = max(aqi, 2)
        
        # ปรับแต่งด้วยความสบายด้านความร้อน (ถ้ามี)
        if 'thermal_discomfort' in row.index:
            discomfort = row['thermal_discomfort']
            if discomfort >= 4:  # เครียดจากความร้อนรุนแรงหรืออันตรายถึงชีวิต
                aqi = max(aqi, 4)
            elif discomfort >= 3:  # เครียดจากความร้อน
                aqi = max(aqi, 3)
            elif discomfort >= 2:  # ระวัง
                aqi = max(aqi, 2)
                
        return aqi
    
    # Apply the function to create AQI class
    data['aqi_class'] = data.apply(assign_aqi_class, axis=1)
    
    # Display distribution of classes
    class_counts = data['aqi_class'].value_counts().sort_index()
    print("\nNumber of samples in each AQI level:")
    for aqi_class, count in class_counts.items():
        aqi_labels = {
            0: "Very Good (0-25)",
            1: "Good (26-37)",
            2: "Moderate (38-50)",
            3: "Unhealthy (51-90)",
            4: "Very Unhealthy (>90)"
        }
        print(f"  Level {aqi_labels[aqi_class]}: {count} samples ({count/len(data)*100:.1f}%)")
    
    return data

# Perform exploratory data analysis
def perform_eda(data, target_column='aqi_class'):
    print("\nPerforming exploratory data analysis (EDA)...")
    
    # List of feature columns (exclude the target column)
    feature_columns = [col for col in data.columns if col != target_column]
    
    # Basic statistics
    print("\nBasic statistics of the data:")
    print(data[feature_columns].describe())
    
    # Setup plots
    plt.figure(figsize=(10, 6))
    
    # Plot distribution of classes
    plt.subplot(2, 2, 1)
    sns.countplot(x=target_column, data=data)
    plt.title('Number of Samples in Each Air Quality Level')
    
    # Plot correlation heatmap (for numerical features)
    plt.subplot(2, 2, 2)
    numeric_columns = data.select_dtypes(include=np.number).columns
    correlation = data[numeric_columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8})
    plt.title('Correlation Between Variables')
    
    # Plot PM2.5 distribution by class
    if 'pm2_5' in data.columns:
        plt.subplot(2, 1, 2)
        sns.boxplot(x=target_column, y='pm2_5', data=data)
        plt.title('PM2.5 Distribution in Each Air Quality Level')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300)
    print("EDA plots saved to 'eda_plots.png'")
    
    return feature_columns

# Train RandomForest model
def train_model(data, feature_columns, target_column='aqi_class'):
    print("\nPreparing data for model training...")
    
    # Prepare features and target
    X = data[feature_columns]
    y = data[target_column]
    
    # Remove direct dependency between PM2.5 and AQI class
    # Remove PM2.5 from model variables since AQI class is directly created from PM2.5
    adjusted_features = [col for col in feature_columns if col != 'pm2_5']
    
    if not adjusted_features:
        print("Warning: Cannot remove 'pm2_5' as it's the only feature. Will add noise to prevent perfect prediction.")
        X = data[feature_columns].copy()
        
        # Add noise to PM2.5 values to prevent over-precise prediction
        print("Adding noise to PM2.5 values to prevent overfitting...")
        np.random.seed(42)  # To ensure same results each time
        X['pm2_5'] = X['pm2_5'] * np.random.uniform(0.95, 1.05, size=len(X))
    else:
        print(f"Removed 'pm2_5' from features to prevent data leakage. Using {len(adjusted_features)} features.")
        X = data[adjusted_features]
    
    # Ensure we use stratified sampling due to imbalanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Training data class distribution: {dict(sorted(pd.Series(y_train).value_counts().items()))}")
    print(f"Testing data class distribution: {dict(sorted(pd.Series(y_test).value_counts().items()))}")
    
    # Create and train RandomForest model
    print("\nTraining RandomForest model...")
    rf_model = RandomForestClassifier(
        n_estimators=150,       # Increase number of trees
        max_depth=8,            # Reduce tree depth to prevent overfitting
        min_samples_split=5,    # Reduce value to allow easier tree branching
        min_samples_leaf=4,     # Set minimum number of samples in leaf
        max_features='sqrt',    # Use square root of number of features
        bootstrap=True,         # Use bootstrap sampling
        oob_score=True,         # Calculate out-of-bag score
        random_state=42,
        class_weight='balanced' # Balance class weights
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Model training completed")
    print(f"Out-of-bag accuracy score: {rf_model.oob_score_:.4f}")
    
    return rf_model, X_train, X_test, y_train, y_test

# Evaluate model performance
def evaluate_model(model, X_test, y_test, X_train, y_train, feature_columns):
    print("\nEvaluating model performance...")
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # If accuracy exceeds 95%, show warning
    if accuracy > 0.95:
        print("\n⚠️ WARNING: Model accuracy is suspiciously high (>{:.0f}%)!".format(accuracy * 100))
        print("This might indicate data leakage or overfitting. Consider these possible causes:")
        print("1. Target variable (AQI class) might be directly derived from features (PM2.5)")
        print("2. Feature correlation might be too strong")
        print("3. Test dataset might be too similar to training data")
        print("4. The problem might be too easy for this model")
    
    # Classification report with more detailed metrics
    print("\nDetailed classification report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    
    # Show average probability for each class
    print("\nAverage prediction probabilities per class:")
    avg_probs = np.mean(y_pred_proba, axis=0)
    for i, prob in enumerate(avg_probs):
        print(f"  Class {i}: {prob:.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Show both regular and normalized confusion matrices
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("Confusion matrix saved to 'confusion_matrix.png'")
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = model.feature_importances_
    features_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=features_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("Feature importance plot saved to 'feature_importance.png'")
    
    # Check for overfitting
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"\nTraining set accuracy: {train_accuracy:.4f}")
    print(f"Testing set accuracy: {accuracy:.4f}")
    print(f"Difference (train-test): {train_accuracy - accuracy:.4f}")
    
    if train_accuracy - accuracy > 0.05:
        print("⚠️ Warning: Potential overfitting detected as training accuracy is much higher than testing accuracy")
    
    return accuracy

# Perform k-fold cross-validation
def perform_cv(model, X, y):
    print("\nPerforming k-fold cross-validation...")
    
    # Use StratifiedKFold instead of KFold for imbalanced class ratio
    from sklearn.model_selection import StratifiedKFold
    
    # Define stratified k-fold cross-validation
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Perform CV with multiple metrics
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    
    # Display results for all metrics
    print(f"Results of {k}-fold Stratified Cross-Validation:")
    print(f"Mean accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"Mean precision: {cv_results['test_precision_macro'].mean():.4f} ± {cv_results['test_precision_macro'].std():.4f}")
    print(f"Mean recall: {cv_results['test_recall_macro'].mean():.4f} ± {cv_results['test_recall_macro'].std():.4f}")
    print(f"Mean F1 score: {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")
    
    # Plot CV scores
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, metric in enumerate(metrics):
        key = f'test_{metric}'
        plt.subplot(2, 2, i+1)
        plt.plot(range(1, k+1), cv_results[key], 'o-', color=colors[i])
        plt.axhline(y=cv_results[key].mean(), color='grey', linestyle='--', 
                   label=f'Mean: {cv_results[key].mean():.4f}')
        plt.xlabel('Fold')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} in {k}-fold CV')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=300)
    print("Cross-validation results plot saved to 'cv_results.png'")
    
    return cv_results['test_accuracy'].mean()

# Save the trained model
def save_model(model, feature_columns):
    print("\nSaving the model...")
    
    # Save model
    joblib.dump(model, 'random_forest_model.joblib')
    
    # Convert pandas Index to list
    feature_list = list(feature_columns) if hasattr(feature_columns, 'tolist') else feature_columns
    
    # Save feature columns
    with open('model_columns.json', 'w') as f:
        json.dump(feature_list, f)
    
    # บันทึก Data Preprocessor
    preprocessor_path = preprocessor.save()
    
    print("Model saved to 'random_forest_model.joblib'")
    print("Feature columns saved to 'model_columns.json'")
    print(f"Data Preprocessor saved to '{preprocessor_path}'")
    
    return 'random_forest_model.joblib', 'model_columns.json', preprocessor_path

# Demo prediction using the trained model
def demo_prediction(model, feature_columns):
    print("\nTesting prediction with the trained model:")
    
    # Example data (you should replace these with realistic values)
    example_data = {
        'temperature': 28.5,
        'humidity': 65.0,
        'pm2_5': 42.0,
        'pm10': 85.0,
        'co': 7.2,
        'o3': 0.058,
        'no2': 0.12,
        'so2': 0.22,
        'co2': 750.0  # Added CO2 as it's now part of our features
    }
    
    # Filter to include only features used in training
    example_filtered = {col: example_data.get(col, 0) for col in feature_columns if col in example_data}
    
    # Check if any features are missing
    missing_features = [col for col in feature_columns if col not in example_data]
    if missing_features:
        print(f"⚠️ Warning: Features {missing_features} not found in sample data, using 0 instead")
        for feat in missing_features:
            example_filtered[feat] = 0
    
    # Create DataFrame
    example_df = pd.DataFrame([example_filtered])
    
    # Ensure all columns are present and in correct order
    for col in feature_columns:
        if col not in example_df.columns:
            example_df[col] = 0
    example_df = example_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(example_df)[0]
    probabilities = model.predict_proba(example_df)[0]
    
    # Display result
    aqi_labels = {
        0: "Very Good (0-25)",
        1: "Good (26-37)",
        2: "Moderate (38-50)",
        3: "Unhealthy (51-90)",
        4: "Very Unhealthy (>90)"
    }
    
    print(f"Sample data:")
    for key, value in example_filtered.items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction result: {prediction} - {aqi_labels[prediction]}")
    print("\nProbability for each class:")
    for i, prob in enumerate(probabilities):
        print(f"  Class {i} ({aqi_labels[i]}): {prob:.4f}")

def main():
    print("=" * 80)
    print("RandomForestClassifier Training Program for Air Quality Analysis")
    print("=" * 80)
    
    # Use the cleaned data file from data_cleaner.py
    file_path = 'expanded_data.csv'  # ใช้ชุดข้อมูลที่มีช่วงอุณหภูมิกว้างขึ้น
    
    print(f"\nUsing expanded temperature range dataset from: {file_path}")
    print("This dataset includes temperatures from 5°C to 45°C for better model generalization.")
    
    try:
        # Load and prepare data
        data = load_dataset(file_path)
        data_cleaned = preprocess_data(data)
        
        # เพิ่มฟีเจอร์ใหม่ด้วย Data Preprocessor
        data_with_features = preprocessor.process_data(data_cleaned)
        
        # สร้างป้าย AQI class โดยพิจารณาอุณหภูมิร่วมด้วย
        data_with_labels = create_aqi_labels(data_with_features)
        
        # Explore data
        feature_columns = perform_eda(data_with_labels)
        
        # Filter out non-feature columns for model training
        excluded_columns = ['aqi_class', 'Report Time', 'timestamp', 'location']
        feature_columns = [col for col in feature_columns if col not in excluded_columns]
        
        # ตรวจสอบการกระจายของคลาส
        if 'aqi_class' in data_with_labels.columns:
            print("\nClass distribution in dataset:")
            class_counts = data_with_labels['aqi_class'].value_counts().sort_index()
            for cls, count in class_counts.items():
                print(f"  Class {cls}: {count} samples ({count/len(data_with_labels)*100:.1f}%)")
            
            # If any class has fewer than 100 samples or fewer than 3 classes
            if class_counts.min() < 100 or len(class_counts) < 3:
                print("\n⚠️ WARNING: Dataset might be imbalanced or lacking diversity!")
                print("Some AQI classes have very few samples or are missing entirely.")
                print("Consider collecting more diverse data or using techniques to handle imbalance.")
        
        print(f"\nUsing the following features for training:")
        for col in feature_columns:
            print(f"  - {col}")
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(data_with_labels, feature_columns)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, X_train, y_train, X_train.columns)
        
        # Cross-validation
        X = data_with_labels[X_train.columns]  # Use single column used in training
        y = data_with_labels['aqi_class']
        perform_cv(model, X, y)
        
        # Save model
        model_path, columns_path, preprocessor_path = save_model(model, X_train.columns)
        
        # Demo prediction
        demo_prediction(model, X_train.columns)
        
        print(f"\nModel successfully trained and saved!")
        print(f"Model path: {model_path}")
        print(f"Columns path: {columns_path}")
        print(f"Preprocessor path: {preprocessor_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main() 