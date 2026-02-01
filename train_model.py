"""
Train a machine learning model to detect namaz from gyroscope data.

This script:
1. Loads the preprocessed data
2. Splits it into training and testing sets
3. Trains a Random Forest classifier
4. Evaluates the model
5. Saves the trained model
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from preprocess import load_and_preprocess_data

def train_model():
    """Train and evaluate the namaz detection model."""
    
    # Create model directory
    Path('models').mkdir(exist_ok=True)
    
    # Load and preprocess data
    print("\n" + "=" * 50)
    print("STEP 1: Loading Data")
    print("=" * 50)
    try:
        X, y, feature_names, subcategories = load_and_preprocess_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Handle class imbalance if necessary
    from collections import Counter
    class_counts = Counter(y)
    print(f"Original class distribution: {class_counts}")

    if any(count < 2 for count in class_counts.values()):
        print("\nWARNING: Some classes have fewer than 2 samples. Stratified split is not possible.")
        print("Duplicating minority samples to enable splitting...")
        # Simple oversampling to fix the immediate crash
        from sklearn.utils import resample
        X_list = [X]
        y_list = [y]
        for class_label, count in class_counts.items():
            if count < 2:
                # Duplicate the minority class samples
                X_minority = X[y == class_label]
                y_minority = y[y == class_label]
                X_upsampled, y_upsampled = resample(X_minority, y_minority,
                                                  replace=True,
                                                  n_samples=2, # Minimum 2 to allow stratified split
                                                  random_state=42)
                # We already have 1, so we just need one more, but resample(n_samples=2) gives 2.
                # To be safe, let's just make sure we have at least 2.
                # Actually, let's just append the missing count.
                X_list.append(X_minority)
                y_list.append(y_minority)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        print(f"New class distribution: {Counter(y)}")
    
    # Split data: 80% training, 20% testing
    print("\n" + "=" * 50)
    print("STEP 2: Splitting Data")
    print("=" * 50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Standardize features (make them have similar scales)
    print("\n" + "=" * 50)
    print("STEP 3: Standardizing Features")
    print("=" * 50)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized")
    
    # Train Random Forest model
    print("\n" + "=" * 50)
    print("STEP 4: Training Model")
    print("=" * 50)
    print("Using Random Forest Classifier...")
    print("  - Number of trees: 100")
    print("  - Max depth: 10")
    
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # Maximum depth of each tree
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ Model training complete!")
    
    # Evaluate on training data
    print("\n" + "=" * 50)
    print("STEP 5: Evaluating Model")
    print("=" * 50)
    
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Evaluate on testing data
    test_predictions = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, test_predictions, 
                                labels=[0, 1, 2, 3],
                                target_names=['Non-Namaz', '2 Rakat', '3 Rakat', '4 Rakat']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2, 3])
    print(f"                 Predicted")
    print(f"                 Non-Namaz  2-Rakat  3-Rakat  4-Rakat")
    print(f"Actual Non-Namaz    {cm[0][0]:4d}     {cm[0][1]:4d}     {cm[0][2]:4d}     {cm[0][3]:4d}")
    print(f"Actual 2-Rakat    {cm[1][0]:4d}     {cm[1][1]:4d}     {cm[1][2]:4d}     {cm[1][3]:4d}")
    print(f"Actual 3-Rakat    {cm[2][0]:4d}     {cm[2][1]:4d}     {cm[2][2]:4d}     {cm[2][3]:4d}")
    print(f"Actual 4-Rakat    {cm[3][0]:4d}     {cm[3][1]:4d}     {cm[3][2]:4d}     {cm[3][3]:4d}")
    
    # Feature importance
    print("\n" + "=" * 50)
    print("Top 10 Most Important Features:")
    print("=" * 50)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:30s}: {row['importance']:.4f}")
    
    # Save the model and scaler
    print("\n" + "=" * 50)
    print("STEP 6: Saving Model")
    print("=" * 50)
    joblib.dump(model, 'models/namaz_detector.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save subcategory mapping for detailed predictions
    unique_subcategories = list(set(subcategories))
    joblib.dump(unique_subcategories, 'models/subcategories.pkl')
    
    print("✓ Model saved to models/namaz_detector.pkl")
    print("✓ Scaler saved to models/scaler.pkl")
    print("✓ Feature names saved to models/feature_names.pkl")
    print("✓ Subcategories saved to models/subcategories.pkl")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print("\nYou can now use predict.py to make predictions on new data.")

if __name__ == "__main__":
    import pandas as pd
    train_model()
