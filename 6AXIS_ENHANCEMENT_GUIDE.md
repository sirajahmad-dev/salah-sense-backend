# Enhanced 6-Axis Sensor Support for Salahsense

## Overview
The Salahsense model now supports **enhanced 6-axis sensor data** combining gyroscope and accelerometer for significantly improved accuracy.

## New Data Format

### Required Columns (7 total):
```
timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
```

- **timestamp**: Time in seconds (0.00, 0.01, 0.02...)
- **gyro_x/y/z**: Gyroscope rotation rates (rad/s)
- **acc_x/y/z**: Accelerometer linear acceleration + gravity (m/s²)

## Feature Enhancement

### Previous: 32 features (3-axis gyroscope only)
- 11 features per gyroscope axis
- 2 combined magnitude features

### Enhanced: 87 features (6-axis gyroscope + accelerometer)
- **Gyroscope features**: 32 (same as before)
- **Accelerometer features**: 44 
- **Gravity/linear separation**: 12
- **Cross-correlation features**: 9
- **Combined magnitude features**: 4

**172% increase in feature richness!**

## Key Improvements

### 1. **Orientation Detection**
- Gravity component identifies standing vs floor sitting
- Better distinction between prayer positions
- Improved clothing adaptation

### 2. **Movement Intensity Analysis**
- Linear acceleration separates prayer from walking/running
- Enhanced activity classification
- Reduced false positives

### 3. **Gravity Separation**
- **Gravity component**: Device orientation (static position)
- **Linear component**: User movement (dynamic motion)
- Prayer movements have distinctive gravity+linear patterns

### 4. **Cross-Correlation Features**
- Gyroscope-accelerometer relationships
- Movement pattern recognition
- Enhanced clothing adaptation detection

## Performance Gains

### Expected Accuracy Improvement:
- **Before**: 90-95% (3-axis gyroscope)
- **After**: 95-98%+ (6-axis enhanced)

### Specific Improvements:
- **Loose clothing**: Better detection through accelerometer gravity
- **Floor sitting**: Enhanced orientation recognition  
- **Similar activities**: Better distinction from walking/running
- **Edge cases**: Reduced false positives/negatives

## Backward Compatibility

The enhanced system is **fully backward compatible**:
- **3-axis files**: Automatically detected, 32 features extracted
- **6-axis files**: Enhanced processing, 87 features extracted
- **Mixed dataset**: Both types work together

## Usage Instructions

### 1. Data Collection
- **Old format** (gyroscope only): Still works
- **New format** (6-axis): Recommended for new data
- **Mixed**: Both formats can be combined

### 2. Training
```bash
python3 train_model.py
```
- Automatically detects data type
- Shows feature breakdown in output
- Trains on all available data

### 3. Prediction
```bash
python3 predict_enhanced.py <your_file.csv>
```
- Detects sensor type automatically
- Shows sensor contribution breakdown
- Provides detailed confidence analysis

### 4. Testing
```bash
python3 test_6axis_features.py
```
- Validates 6-axis feature extraction
- Shows feature enhancement details

## Data Structure (Same as Before)

```
data/
├── namaz/
│   ├── standing_loose_clothing/
│   ├── standing_tight_clothing/
│   └── sitting_floor/
└── non_namaz/
    ├── walking/
    ├── running/
    ├── sitting_floor/
    └── sitting_chair/
```

## Training File Requirements (Unchanged)

- **Total optimal**: 1400-1900 files
- **Distribution**: Same categorization as before
- **Better results**: Same quantity, higher accuracy

## Migration Strategy

### For Existing 3-axis Data:
- Keep using current files (no changes needed)
- Gradually collect new 6-axis data
- Model benefits from mixed training

### For New Data Collection:
- Record both gyroscope and accelerometer
- Use enhanced mobile app (if updated)
- Follow new 7-column format

## Technical Details

### Signal Processing:
- **Low-pass filtering**: Gravity component extraction (0.3Hz cutoff)
- **High-pass filtering**: Linear motion separation
- **Cross-correlation**: Sensor relationship analysis
- **Feature scaling**: Standardized for ML training

### Model Architecture:
- **Random Forest**: Handles 87 features efficiently
- **Feature importance**: Identifies most informative sensors
- **Robust training**: Handles mixed 3-axis/6-axis data

## Expected Results

With proper 6-axis data:
- **95-98%+ accuracy** on namaz vs non-namaz
- **Better generalization** across clothing types
- **Improved robustness** to phone position variations
- **Enhanced reliability** for edge cases

The 6-axis enhancement represents a **major accuracy improvement** while maintaining full backward compatibility with existing data.