"""
Enhanced prediction script that can identify specific subcategories.
"""
import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path

def load_models():
    """Load the trained model, scaler, and metadata."""
    model = joblib.load('models/namaz_detector.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Try to load subcategories if available
    try:
        subcategories = joblib.load('models/subcategories.pkl')
        has_subcategories = True
    except:
        subcategories = []
        has_subcategories = False
    
    return model, scaler, feature_names, subcategories, has_subcategories

def preprocess_single_file(file_path, feature_names):
    """Preprocess a single CSV file for prediction."""
    df = pd.read_csv(file_path)
    
    # Import the feature extraction function
    from preprocess import extract_features
    features = extract_features(df)
    
    # Convert to DataFrame to ensure correct feature order
    features_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    for feature_name in feature_names:
        if feature_name not in features_df.columns:
            features_df[feature_name] = 0
    
    # Reorder columns to match training
    features_df = features_df[feature_names]
    
    return features_df.values[0]

def predict_category(file_path):
    """Predict the category and subcategory for a given file."""
    
    # Load models
    model, scaler, feature_names, subcategories, has_subcategories = load_models()
    
    # Preprocess the file
    features = preprocess_single_file(file_path, feature_names)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    prediction_proba = model.predict_proba(features_scaled)[0]
    
    # Check if we have accelerometer features
    has_accel_features = any('acc_' in feature for feature in feature_names)
    
    # Map prediction to meaningful label
    prediction_map = {
        0: "Non-Namaz",
        1: "2 Rakat",
        2: "3 Rakat", 
        3: "4 Rakat"
    }
    
    predicted_label = prediction_map.get(prediction, "Unknown")
    
    # Output results
    print("\n" + "=" * 50)
    print("RAKAT DETECTION RESULTS")
    print("=" * 50)
    print(f"File: {file_path}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {max(prediction_proba) * 100:.2f}%")
    
    # Show probability breakdown for all classes
    class_names = ['Non-Namaz', '2 Rakat', '3 Rakat', '4 Rakat']
    print("\nProbability Breakdown:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {prediction_proba[i] * 100:.2f}%")
    
    if has_accel_features:
        print(f"\nData Type: Enhanced 6-axis (Gyro + Accelerometer)")
        acc_features_count = len([f for f in feature_names if 'acc_' in f])
        gyro_features_count = len([f for f in feature_names if 'gyro_' in f])
        rakat_features_count = len([f for f in feature_names if any(keyword in f for keyword in ['peak', 'duration', 'segment', 'entropy', 'rakat'])])
        print(f"Sensors Used: {acc_features_count} accel + {gyro_features_count} gyro features")
        print(f"Rakat Features: {rakat_features_count} cycle detection features")
    else:
        print(f"\nData Type: 3-axis Gyroscope only")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 contributing features:")
    for idx, row in feature_importance.head(5).iterrows():
        sensor_type = "[Rakat]" if any(keyword in row['feature'] for keyword in ['peak', 'duration', 'segment', 'entropy', 'rakat']) else "[Accel]" if 'acc_' in row['feature'] else "[Gyro]" if 'gyro_' in row['feature'] else "[Combined]"
        print(f"  {row['feature']}: {row['importance']:.4f} {sensor_type}")
    
    # Show feature type breakdown
    if has_accel_features:
        rakat_importance = feature_importance[feature_importance['feature'].str.contains('peak|duration|segment|entropy|rakat', case=False)]['importance'].sum()
        accel_importance = feature_importance[feature_importance['feature'].str.contains('acc_')]['importance'].sum()
        gyro_importance = feature_importance[feature_importance['feature'].str.contains('gyro_')]['importance'].sum()
        combined_importance = feature_importance[feature_importance['feature'].str.contains('magnitude|total|_corr', case=False)]['importance'].sum()
        
        print("\nFeature Type Analysis:")
        print(f"  Rakat Detection: {rakat_importance*100:.1f}%")
        print(f"  Accelerometer: {accel_importance*100:.1f}%")
        print(f"  Gyroscope: {gyro_importance*100:.1f}%")
        print(f"  Combined: {combined_importance*100:.1f}%")
    
    # Show confidence interpretation
    confidence = max(prediction_proba)
    if confidence > 0.85:
        confidence_level = "High Confidence"
    elif confidence > 0.70:
        confidence_level = "Medium Confidence"
    else:
        confidence_level = "Low Confidence"
    
    print(f"\nConfidence Level: {confidence_level}")
    
    if predicted_label != "Non-Namaz":
        print(f"\n✓ Rakat Count Predicted: {predicted_label}")
    else:
        print("\n✗ No Namaz Detected")
    
    return prediction, prediction_proba

def main():
    parser = argparse.ArgumentParser(description='Predict namaz category from 6-axis sensor data')
    parser.add_argument('file', help='Path to the CSV file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File '{args.file}' not found.")
        return
    
    # Check if file is CSV
    if not args.file.lower().endswith('.csv'):
        print(f"Error: Only CSV files are supported.")
        print(f"Please provide a .csv file.")
        return
    
    try:
        prediction, probability = predict_category(args.file)
        
        if prediction == 1:
            print("\n✓ This appears to be NAMAZ (prayer) movement")
        else:
            print("\n✗ This does not appear to be namaz movement")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Please ensure the CSV file has the correct format:")
        print("Columns: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z")

if __name__ == "__main__":
    main()