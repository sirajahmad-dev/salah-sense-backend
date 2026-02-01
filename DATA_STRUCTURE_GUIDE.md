# Enhanced Data Structure for Salahsense

## Overview
The model now supports **subcategorized data** to improve accuracy for different prayer styles and clothing types.

## New Data Structure

```
data/
├── namaz/
│   ├── standing_loose_clothing/    # Namaz with kurta, thawb, etc.
│   ├── standing_tight_clothing/    # Namaz with shirt, jeans, etc.
│   └── sitting_floor/              # Floor-based sitting namaz
└── non_namaz/
    ├── walking/                     # Walking activities
    ├── running/                     # Running activities  
    ├── sitting_floor/              # Sitting on floor (not praying)
    └── sitting_chair/              # Sitting on chair
```

## Training Data Requirements

### Minimum Files for Good Performance:
- **Standing namaz (loose clothing)**: 200-300 files
- **Standing namaz (tight clothing)**: 200-300 files  
- **Floor sitting namaz**: 200-300 files
- **Non-namaz activities**: 800-1000 files
  - Walking: 200-300 files
  - Running: 200-300 files
  - Floor sitting: 200-300 files
  - Chair sitting: 100-200 files

**Total optimal: 1400-1900 files**

## Usage Instructions

### 1. Organize Your Data
Place your Excel/CSV files in the appropriate subfolders based on:
- **Prayer type**: standing vs floor sitting
- **Clothing**: loose (kurta/thawb) vs tight (shirt/jeans)
- **Activity type**: walking/running/sitting

### 2. Train the Model
```bash
python3 train_model.py
```

The enhanced training will:
- Load data from all subfolders
- Display file counts per category
- Train on the combined dataset
- Save subcategory information

### 3. Make Predictions
```bash
python3 predict_enhanced.py your_file.csv
```

Enhanced prediction shows:
- **Primary classification**: Namaz vs Non-namaz
- **Confidence percentage**
- **Top contributing features**
- **Detailed probability breakdown**

## File Format Requirements

All files must contain these columns:
- `timestamp`: Time in seconds (0.00, 0.01, 0.02...)
- `gyro_x`: X-axis gyroscope reading
- `gyro_y`: Y-axis gyroscope reading  
- `gyro_z`: Z-axis gyroscope reading

Example:
```
timestamp,gyro_x,gyro_y,gyro_z
0.000,-0.177,0.216,-0.137
0.010,-0.244,-0.468,-0.277
0.020,-0.154,-0.228,-0.277
...
```

## Why This Structure Improves Accuracy

1. **Clothing Impact**: Loose clothing dampens movement patterns
2. **Prayer Styles**: Standing vs floor-based prayers have different movement signatures
3. **Activity Separation**: Different non-namaz activities have distinct patterns
4. **Targeted Learning**: Model learns specific patterns for each scenario

## Reorganizing Existing Data

If you have existing data, use:
```bash
./setup_data_structure.sh
```

This creates the folder structure and moves existing files to temporary folders (`temp_namaz_files/`, `temp_non_namaz_files/`) for manual reorganization.

## Expected Performance

With properly organized data:
- **90-95%+ accuracy** on namaz vs non-namaz classification
- **Better distinction** between different prayer styles
- **Reduced false positives** from similar activities
- **Improved clothing adaptation**

## Next Steps

1. **Collect diverse data** covering all categories
2. **Ensure balanced distribution** across subcategories  
3. **Train with enhanced script** for optimal results
4. **Test with new samples** to validate performance