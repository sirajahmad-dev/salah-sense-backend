#!/bin/bash
# Quick start script for Enhanced 6-Axis Namaz Detection System

echo "=========================================="
echo "Enhanced 6-Axis Namaz Detection - Quick Start"
echo "=========================================="
echo ""
echo "Note: This system supports CSV files only (6-axis sensor data)"
echo ""

# Step 1: Setup data structure
echo "Step 1/3: Setting up data structure..."
if [ ! -d "data" ]; then
    ./setup_data_structure.sh
else
    echo "Data structure already exists"
fi
echo ""

# Step 2: Train the model (if data exists)
echo "Step 2/3: Training the model..."
if [ -d "data" ] && [ "$(find data -name "*.csv" | wc -l)" -gt 0 ]; then
    python3 train_model.py
else
    echo "No CSV data found. Please organize your CSV files first:"
    echo "  data/namaz/standing_loose_clothing/"
    echo "  data/namaz/standing_tight_clothing/"
    echo "  data/namaz/sitting_floor/"
    echo "  data/non_namaz/walking/"
    echo "  data/non_namaz/running/"
    echo "  data/non_namaz/sitting_floor/"
    echo "  data/non_namaz/sitting_chair/"
fi
echo ""

# Step 3: Test predictions (if model exists)
echo "Step 3/3: Testing predictions..."
if [ -f "models/namaz_detector.pkl" ]; then
    echo "To test with your CSV files:"
    echo "  python3 predict_enhanced.py your_namaz_file.csv"
    echo "  python3 predict_enhanced.py your_non_namaz_file.csv"
else
    echo "No trained model found. Please train the model first."
fi
echo ""

echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "CSV File Format Required:"
echo "timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z"
echo ""
echo "For detailed documentation:"
echo "- README.md: Complete guide"
echo "- 6AXIS_ENHANCEMENT_GUIDE.md: Technical details"
echo "- DATA_STRUCTURE_GUIDE.md: Data organization"
echo ""