"""
Reorganize existing data into the new subfolder structure.
This script helps organize current data files into appropriate subcategories.
"""
import os
import shutil
from pathlib import Path
import pandas as pd

def get_user_category_choice(file_path):
    """Ask user to categorize a CSV file."""
    print(f"\nCSV File: {file_path.name}")
    print("Please categorize this file:")
    
    print("\nNAMAZ CATEGORIES:")
    print("1. Standing + Loose Clothing (kurta, thawb)")
    print("2. Standing + Tight Clothing (shirt, jeans)")
    print("3. Sitting Floor (floor-based prayer)")
    
    print("\nNON-NAMAZ CATEGORIES:")
    print("4. Walking")
    print("5. Running")
    print("6. Sitting Floor (not praying)")
    print("7. Sitting Chair")
    print("8. Other/Skip")
    
    while True:
        try:
            choice = int(input("Enter category number (1-8): "))
            if 1 <= choice <= 8:
                return choice
            else:
                print("Please enter a number between 1-8")
        except ValueError:
            print("Please enter a valid number")

def create_data_structure():
    """Create the new folder structure."""
    base_path = Path('data')
    
    # Create namaz subfolders
    (base_path / 'namaz' / 'standing_loose_clothing').mkdir(parents=True, exist_ok=True)
    (base_path / 'namaz' / 'standing_tight_clothing').mkdir(parents=True, exist_ok=True)
    (base_path / 'namaz' / 'sitting_floor').mkdir(parents=True, exist_ok=True)
    
    # Create non-namaz subfolders
    (base_path / 'non_namaz' / 'walking').mkdir(parents=True, exist_ok=True)
    (base_path / 'non_namaz' / 'running').mkdir(parents=True, exist_ok=True)
    (base_path / 'non_namaz' / 'sitting_floor').mkdir(parents=True, exist_ok=True)
    (base_path / 'non_namaz' / 'sitting_chair').mkdir(parents=True, exist_ok=True)
    
    print("✓ Folder structure created")

def move_file_to_category(file_path, choice):
    """Move file to appropriate category folder."""
    
    category_map = {
        1: ('namaz', 'standing_loose_clothing'),
        2: ('namaz', 'standing_tight_clothing'),
        3: ('namaz', 'sitting_floor'),
        4: ('non_namaz', 'walking'),
        5: ('non_namaz', 'running'),
        6: ('non_namaz', 'sitting_floor'),
        7: ('non_namaz', 'sitting_chair')
    }
    
    if choice == 8:
        print(f"Skipping {file_path.name}")
        return
    
    main_folder, subfolder = category_map[choice]
    destination = Path('data') / main_folder / subfolder / file_path.name
    
    # Handle filename conflicts
    counter = 1
    original_dest = destination
    while destination.exists():
        stem = original_dest.stem
        suffix = original_dest.suffix
        destination = original_dest.parent / f"{stem}_{counter}{suffix}"
        counter += 1
    
    shutil.move(str(file_path), str(destination))
    print(f"Moved {file_path.name} → {main_folder}/{subfolder}/")

def auto_categorize_by_filename(file_path):
    """Auto-categorize based on filename patterns."""
    name_lower = file_path.name.lower()
    
    # Auto-detect namaz patterns
    if any(keyword in name_lower for keyword in ['namaz', 'rakat', 'prayer']):
        if any(keyword in name_lower for keyword in ['sitting', 'chair', 'floor']):
            return 3  # sitting floor namaz
        elif any(keyword in name_lower for keyword in ['kurta', 'thawb', 'loose']):
            return 1  # loose clothing
        elif any(keyword in name_lower for keyword in ['shirt', 'jean', 'tight']):
            return 2  # tight clothing
        else:
            return None  # ask user
    
    # Auto-detect non-namaz patterns
    elif 'walking' in name_lower:
        return 4
    elif 'running' in name_lower or 'run' in name_lower:
        return 5
    elif 'sitting' in name_lower:
        return 6  # assume floor sitting for now
    else:
        return None  # ask user

def reorganize_data():
    """Main function to reorganize data."""
    
    # Create folder structure
    create_data_structure()
    
    # Process existing files
    data_path = Path('data')
    
    # Process namaz files
    namaz_path = data_path / 'namaz'
    if namaz_path.exists():
        namaz_files = [f for f in namaz_path.iterdir() if f.is_file() and f.suffix == '.csv']
        
        for file_path in namaz_files:
            # Skip if already in subfolder
            if file_path.parent != namaz_path:
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing CSV: {file_path.name}")
            
            # Try auto-categorization first
            auto_choice = auto_categorize_by_filename(file_path)
            
            if auto_choice:
                print(f"Auto-detected category: {auto_choice}")
                confirm = input("Use auto-detection? (y/n): ").lower()
                if confirm == 'y':
                    move_file_to_category(file_path, auto_choice)
                    continue
            
            # Manual categorization
            choice = get_user_category_choice(file_path)
            move_file_to_category(file_path, choice)
    
    # Process non-namaz files
    non_namaz_path = data_path / 'non_namaz'
    if non_namaz_path.exists():
        non_namaz_files = [f for f in non_namaz_path.iterdir() if f.is_file() and f.suffix == '.csv']
        
        for file_path in non_namaz_files:
            # Skip if already in subfolder
            if file_path.parent != non_namaz_path:
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing CSV: {file_path.name}")
            
            # Try auto-categorization first
            auto_choice = auto_categorize_by_filename(file_path)
            
            if auto_choice:
                print(f"Auto-detected category: {auto_choice}")
                confirm = input("Use auto-detection? (y/n): ").lower()
                if confirm == 'y':
                    move_file_to_category(file_path, auto_choice)
                    continue
            
            # Manual categorization
            choice = get_user_category_choice(file_path)
            move_file_to_category(file_path, choice)
    
    print("\n" + "="*50)
    print("✓ Data reorganization complete!")
    
    # Show new structure
    print("\nNew data structure:")
    for main_folder in ['namaz', 'non_namaz']:
        main_path = data_path / main_folder
        if main_path.exists():
            print(f"\n{main_folder.upper()}/")
            for subfolder in main_path.iterdir():
                if subfolder.is_dir():
                    files = len([f for f in subfolder.iterdir() if f.suffix == '.csv'])
                    print(f"  {subfolder.name}/: {files} CSV files")

if __name__ == "__main__":
    print("CSV Data Reorganization Tool")
    print("This will move existing CSV files into categorized subfolders.")
    print("\nRecommended folder structure:")
    print("data/namaz/")
    print("  ├── standing_loose_clothing/")
    print("  ├── standing_tight_clothing/")
    print("  └── sitting_floor/")
    print("data/non_namaz/")
    print("  ├── walking/")
    print("  ├── running/")
    print("  ├── sitting_floor/")
    print("  └── sitting_chair/")

    proceed = input("\nProceed with reorganization? (y/n): ").lower()
    if proceed == 'y':
        reorganize_data()
    else:
        print("Reorganization cancelled.")