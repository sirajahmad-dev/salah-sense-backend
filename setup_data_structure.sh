#!/bin/bash
# Script to create the new data structure and move existing files

echo "Creating new data structure with rakat categories..."

# Create namaz subfolders for each rakat count
for rakat in 2_rakat 3_rakat 4_rakat; do
    mkdir -p data/namaz/$rakat/standing_loose_clothing
    mkdir -p data/namaz/$rakat/standing_tight_clothing  
    mkdir -p data/namaz/$rakat/sitting_floor
done

# Create non-namaz subfolders
mkdir -p data/non_namaz/walking
mkdir -p data/non_namaz/running
mkdir -p data/non_namaz/sitting_floor
mkdir -p data/non_namaz/sitting_chair

echo "Folder structure with rakat categories created!"

echo ""
echo "Move your CSV files manually to the appropriate subfolders:"
echo ""
echo "For NAMAZ files by RAKAT count:"
echo "  data/namaz/2_rakat/standing_loose_clothing/  - 2 rakat namaz, loose clothing"
echo "  data/namaz/2_rakat/standing_tight_clothing/  - 2 rakat namaz, tight clothing"
echo "  data/namaz/2_rakat/sitting_floor/            - 2 rakat floor sitting namaz"
echo ""
echo "  data/namaz/3_rakat/standing_loose_clothing/  - 3 rakat namaz, loose clothing"
echo "  data/namaz/3_rakat/standing_tight_clothing/  - 3 rakat namaz, tight clothing"
echo "  data/namaz/3_rakat/sitting_floor/            - 3 rakat floor sitting namaz"
echo ""
echo "  data/namaz/4_rakat/standing_loose_clothing/  - 4 rakat namaz, loose clothing"
echo "  data/namaz/4_rakat/standing_tight_clothing/  - 4 rakat namaz, tight clothing"
echo "  data/namaz/4_rakat/sitting_floor/            - 4 rakat floor sitting namaz"
echo ""
echo "For NON-NAMAZ files:"
echo "  data/non_namaz/walking/              - Walking activities"
echo "  data/non_namaz/running/              - Running activities"
echo "  data/non_namaz/sitting_floor/        - Sitting on floor (not praying)"
echo "  data/non_namaz/sitting_chair/        - Sitting on chair"
echo ""

# Move existing files to temp location for manual organization
if [ -d "data/namaz" ]; then
    mkdir -p temp_namaz_files
    find data/namaz -maxdepth 1 -name "*.csv" -exec mv {} temp_namaz_files/ \;
    echo "Moved existing namaz CSV files to temp_namaz_files/ for reorganization"
fi

if [ -d "data/non_namaz" ]; then
    mkdir -p temp_non_namaz_files  
    find data/non_namaz -maxdepth 1 -name "*.csv" -exec mv {} temp_non_namaz_files/ \;
    echo "Moved existing non-namaz CSV files to temp_non_namaz_files/ for reorganization"
fi

echo ""
echo "After organizing your files, run:"
echo "  python3 train_model.py"
echo "  python3 predict_enhanced.py <your_file.csv>"