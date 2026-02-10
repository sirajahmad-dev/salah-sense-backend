"""
FastAPI interface for Salahsense Namaz Detection System.

This API provides endpoints for:
1. Uploading CSV training data to appropriate subfolders
2. Training the model on uploaded data
3. Making predictions on new CSV files
4. Getting system status and statistics
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import json
import subprocess
from typing import Optional, List
import asyncio
import io
import base64
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from scipy import signal
import numpy.fft as fft


app = FastAPI(
    title="Salahsense API",
    description="Enhanced 6-axis Namaz Detection System API",
    version="1.0.0"
)

# Enable CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for mobile app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
UPLOAD_DIR = Path(__file__).parent / "uploads"

# Valid subcategories
VALID_NAMAZ_SUBCATEGORIES = [
    "standing_loose_clothing",
    "standing_tight_clothing", 
    "sitting_floor"
]

VALID_NON_NAMAZ_CATEGORIES = [
    "walking",
    "running",
    "sitting_floor",
    "sitting_chair"
]

VALID_RAKAT_COUNTS = ["2_rakat", "3_rakat", "4_rakat"]

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Create rakat-specific namaz directories
    for rakat_count in VALID_RAKAT_COUNTS:
        for subcategory in VALID_NAMAZ_SUBCATEGORIES:
            (DATA_DIR / "namaz" / rakat_count / subcategory).mkdir(parents=True, exist_ok=True)
    
    # Create non-namaz directories
    for category in VALID_NON_NAMAZ_CATEGORIES:
        (DATA_DIR / "non_namaz" / category).mkdir(parents=True, exist_ok=True)

def validate_csv_content(df):
    """Validate that CSV has required columns and format."""
    required_columns = ["timestamp", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing_columns}"
        )
    
    # Check for numeric data
    numeric_columns = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Column '{col}' contains non-numeric data"
                    )
            except:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{col}' cannot be converted to numeric"
                )
    
    # Check minimum data length
    if len(df) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have at least 50 rows of data. Found: {len(df)}"
        )
    
    return True

def get_data_statistics():
    """Get current data statistics."""
    stats = {
        "namaz": {
            "2_rakat": {},
            "3_rakat": {}, 
            "4_rakat": {}
        },
        "non_namaz": {},
        "total_files": 0,
        "categories": {}
    }
    
    # Count namaz files by rakat count
    for rakat_count in VALID_RAKAT_COUNTS:
        for subcategory in VALID_NAMAZ_SUBCATEGORIES:
            category_path = DATA_DIR / "namaz" / rakat_count / subcategory
            if category_path.exists():
                files = len([f for f in category_path.iterdir() if f.suffix == '.csv'])
                if subcategory not in stats["namaz"][rakat_count]:
                    stats["namaz"][rakat_count][subcategory] = 0
                stats["namaz"][rakat_count][subcategory] = files
                stats["total_files"] += files
    
    # Count non-namaz files  
    for category in VALID_NON_NAMAZ_CATEGORIES:
        category_path = DATA_DIR / "non_namaz" / category
        if category_path.exists():
            files = len([f for f in category_path.iterdir() if f.suffix == '.csv'])
            stats["non_namaz"][category] = files
            stats["total_files"] += files
    
    # Calculate totals
    stats["total_namaz"] = sum(
        sum(rakat_data.values()) for rakat_data in stats["namaz"].values()
    )
    stats["total_non_namaz"] = sum(stats["non_namaz"].values())
    
    # Calculate rakat-specific totals
    for rakat_count in VALID_RAKAT_COUNTS:
        stats["namaz"][rakat_count]["total"] = sum(stats["namaz"][rakat_count].values())
    
    return stats

def get_files_in_directory(directory_path: Path) -> List[dict]:
    """Get list of CSV files in a directory with metadata."""
    files = []
    if not directory_path.exists():
        return files
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix == '.csv':
            try:
                # Get file size and modification time
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path.relative_to(DATA_DIR))
                })
            except Exception:
                continue
    
    return sorted(files, key=lambda x: x["name"])

def get_subdirectories(directory_path: Path) -> List[dict]:
    """Get list of subdirectories with cumulative file counts."""
    subdirs = []
    if not directory_path.exists():
        return subdirs
    
    for subdir_path in directory_path.iterdir():
        if subdir_path.is_dir():
            # Count CSV files in this subdirectory and all its subdirectories (cumulative)
            csv_files = len([f for f in subdir_path.rglob("*.csv") if f.is_file()])
            subdirs.append({
                "name": subdir_path.name,
                "file_count": csv_files,
                "path": str(subdir_path.relative_to(DATA_DIR))
            })
    
    return sorted(subdirs, key=lambda x: x["name"])

def count_csv_files_in_directory(directory_path: Path) -> int:
    """Count all CSV files in a directory and its subdirectories."""
    if not directory_path.exists():
        return 0
    
    return len([f for f in directory_path.rglob("*.csv") if f.is_file()])

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    ensure_directories()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Salahsense API - Enhanced 6-Axis Namaz Detection",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "train": "/train", 
            "predict": "/predict",
            "status": "/status",
            "categories": "/categories",
            "data": "/data",
            "data_listing": "/data/{path}",
            "visualize": "/visualize/{path}",
            "visualizer": "/visualizer"
        },
        "supported_format": "CSV with 7 columns: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z"
    }

@app.get("/categories")
async def get_categories():
    """Get available data categories."""
    return {
        "namaz": {
            "subcategories": VALID_NAMAZ_SUBCATEGORIES,
            "rakat_counts": VALID_RAKAT_COUNTS
        },
        "non_namaz": VALID_NON_NAMAZ_CATEGORIES
    }

@app.get("/status")
async def get_status():
    """Get current system status and data statistics."""
    stats = get_data_statistics()
    
    # Check if model exists
    model_exists = (MODELS_DIR / "namaz_detector.pkl").exists()
    
    return {
        "status": "ready",
        "model_trained": model_exists,
        "data_statistics": stats,
        "ready_for_training": stats["total_files"] > 0,
        "recommendation": {
            "min_files_per_category": 50,
            "optimal_files_per_category": 200,
            "total_optimal_files": 1400
        }
    }

@app.post("/upload")
async def upload_csv_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category_type: str = Form(...),
    category: str = Form(...),
    rakat_count: Optional[str] = Form(None),
    name: Optional[str] = Form(None)
):
    """
    Upload a CSV file to appropriate data category.
    
    Args:
        file: CSV file to upload
        category_type: "namaz" or "non_namaz"
        category: Specific subcategory name
        rakat_count: "2_rakat", "3_rakat", "4_rakat" (for namaz only)
        name: Custom filename (without extension) - saves as {name}_rakat_date_time.csv
    """
    
    # Validate category type
    if category_type not in ["namaz", "non_namaz"]:
        raise HTTPException(
            status_code=400,
            detail="category_type must be 'namaz' or 'non_namaz'"
        )
    
    # Handle namaz upload with rakat count
    if category_type == "namaz":
        # Validate rakat count
        if not rakat_count or rakat_count not in VALID_RAKAT_COUNTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rakat count. Valid options: {VALID_RAKAT_COUNTS}"
            )
        
        # Validate subcategory
        if category not in VALID_NAMAZ_SUBCATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid namaz subcategory. Valid options: {VALID_NAMAZ_SUBCATEGORIES}"
            )
        
        target_dir = DATA_DIR / "namaz" / rakat_count / category
    else:
        # Validate non-namaz category
        if category not in VALID_NON_NAMAZ_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid non_namaz category. Valid options: {VALID_NON_NAMAZ_CATEGORIES}"
            )
        target_dir = DATA_DIR / "non_namaz" / category
    
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Read and validate CSV content
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        validate_csv_content(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(e)}"
        )
    
    # Generate custom filename or use default naming
    if name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if category_type == "namaz" and rakat_count:
            # Namaz naming: name_rakat_category_date_time.csv
            filename = f"{name}_{rakat_count}_{category}_{timestamp}.csv"
        else:
            # Non-namaz naming: name_category_date_time.csv
            filename = f"{name}_{category}_{timestamp}.csv"
    else:
        # Default naming: timestamp_random_original.csv
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file_id}_{file.filename}"
    
    file_path = target_dir / filename
    
    # Save file
    try:
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    return {
        "success": True,
        "message": "File uploaded successfully",
        "file_info": {
            "filename": filename,
            "category_type": category_type,
            "category": category,
            "rakat_count": rakat_count if category_type == "namaz" else None,
            "custom_name": name if name else None,
            "path": str(file_path),
            "rows": len(df),
            "columns": len(df.columns)
        },
        "data_statistics": get_data_statistics()
    }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the namaz detection model on uploaded data.
    
    This is a long-running operation. The API returns immediately and 
    training continues in the background.
    """
    
    stats = get_data_statistics()
    if stats["total_files"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No training data available. Please upload CSV files first."
        )
    
    # Create a training job ID
    job_id = str(uuid.uuid4())[:8]
    log_file = UPLOAD_DIR / f"training_{job_id}.log"
    
    def train_background():
        """Background training function."""
        try:
            # Run training script
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ['python3', 'train_model.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                if process.stdout:
                    for line in process.stdout:
                        f.write(f"{datetime.now()}: {line}")
                        f.flush()
                
                process.wait()
                
                if process.returncode == 0:
                    f.write(f"{datetime.now()}: Training completed successfully")
                else:
                    f.write(f"{datetime.now()}: Training failed with code {process.returncode}")
                    
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()}: Training error: {str(e)}")
    
    # Start background training
    background_tasks.add_task(train_background)
    
    return {
        "success": True,
        "message": "Training started in background",
        "job_id": job_id,
        "log_file": str(log_file),
        "current_data_stats": stats,
        "check_status_endpoint": f"/train_status/{job_id}"
    }

@app.get("/train_status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    log_file = UPLOAD_DIR / f"training_{job_id}.log"
    
    if not log_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Training job not found"
        )
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Determine status from log content
        if "Training completed successfully" in log_content:
            status = "completed"
        elif "Training failed" in log_content:
            status = "failed"
        else:
            status = "running"
        
        return {
            "job_id": job_id,
            "status": status,
            "log": log_content,
            "model_exists": (MODELS_DIR / "namaz_detector.pkl").exists()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read training log: {str(e)}"
        )

@app.post("/predict")
async def predict_namaz(file: UploadFile = File(...)):
    """
    Make a prediction on a CSV file.
    
    Returns the classification result with confidence scores.
    """
    
    # Check if model exists
    if not (MODELS_DIR / "namaz_detector.pkl").exists():
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    # Validate file type
    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    # Save uploaded file temporarily
    temp_file = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.csv"
    
    try:
        contents = await file.read()
        with open(temp_file, 'wb') as f:
            f.write(contents)
        
        # Run prediction script
        result = subprocess.run(
            ['python3', 'predict_enhanced.py', str(temp_file)],
            capture_output=True,
            text=True
        )
        
        # Parse prediction output
        if result.returncode == 0:
            # Extract prediction from output
            output_lines = result.stdout.split('\n')
            
            # Find confidence and prediction
            confidence = None
            prediction = None
            prob_namaz = None
            prob_non_namaz = None
            
            for line in output_lines:
                if "Confidence:" in line:
                    confidence = float(line.split(':')[1].strip().replace('%', ''))
                elif "Prediction:" in line:
                    prediction = line.split(':')[1].strip()
                elif "Probability - Namaz:" in line:
                    prob_namaz = float(line.split(':')[1].strip().replace('%', ''))
                elif "Probability - Non-Namaz:" in line:
                    prob_non_namaz = float(line.split(':')[1].strip().replace('%', ''))
            
            # Clean up temp file
            temp_file.unlink()
            
            return {
                "success": True,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "namaz": prob_namaz,
                    "non_namaz": prob_non_namaz
                },
                "file_info": {
                    "filename": file.filename,
                    "rows": len(pd.read_csv(str(temp_file))),
                    "processed_at": datetime.now().isoformat()
                }
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.stderr}"
            )
            
    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.delete("/data/{category_type}/{category}/{filename}")
async def delete_data_file(category_type: str, category: str, filename: str):
    """Delete a specific data file."""
    
    # Validate paths
    if category_type not in ["namaz", "non_namaz"]:
        raise HTTPException(status_code=400, detail="Invalid category type")
    
    file_path = DATA_DIR / category_type / category / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return {
            "success": True,
            "message": f"File {filename} deleted successfully",
            "updated_stats": get_data_statistics()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )

@app.get("/download/model")
async def download_model():
    """Download the trained model."""
    model_file = MODELS_DIR / "namaz_detector.pkl"
    
    if not model_file.exists():
        raise HTTPException(
            status_code=404,
            detail="No trained model found"
        )
    
    return FileResponse(
        model_file,
        media_type="application/octet-stream",
        filename="namaz_detector.pkl"
    )

@app.get("/data")
async def get_data_root():
    """Get listing of all data in root data folder."""
    result = {
        "path": "data",
        "type": "root",
        "categories": {},
        "total_files": 0
    }
    
    # List namaz directory
    namaz_path = DATA_DIR / "namaz"
    if namaz_path.exists():
        namaz_subdirs = get_subdirectories(namaz_path)
        namaz_total = count_csv_files_in_directory(namaz_path)
        result["categories"]["namaz"] = {
            "type": "category",
            "subcategories": namaz_subdirs,
            "total_files": namaz_total
        }
        result["total_files"] += namaz_total
    
    # List non_namaz directory
    non_namaz_path = DATA_DIR / "non_namaz"
    if non_namaz_path.exists():
        non_namaz_subdirs = get_subdirectories(non_namaz_path)
        non_namaz_total = count_csv_files_in_directory(non_namaz_path)
        result["categories"]["non_namaz"] = {
            "type": "category",
            "subcategories": non_namaz_subdirs,
            "total_files": non_namaz_total
        }
        result["total_files"] += non_namaz_total
    
    return result

@app.get("/data/{category}")
async def get_data_category(category: str):
    """Get listing of files in a category (namaz or non_namaz)."""
    if category not in ["namaz", "non_namaz"]:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available: namaz, non_namaz"
        )
    
    category_path = DATA_DIR / category
    if not category_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Category directory '{category}' not found"
        )
    
    result = {
        "path": f"data/{category}",
        "type": "category",
        "name": category,
        "subcategories": get_subdirectories(category_path),
        "files": get_files_in_directory(category_path),
        "total_files": count_csv_files_in_directory(category_path)
    }
    
    return result

@app.get("/data/namaz/{rakat_count}")
async def get_namaz_rakat(rakat_count: str):
    """Get listing of files in a namaz rakat count directory."""
    if rakat_count not in VALID_RAKAT_COUNTS:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid rakat count '{rakat_count}'. Available: {VALID_RAKAT_COUNTS}"
        )
    
    rakat_path = DATA_DIR / "namaz" / rakat_count
    if not rakat_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Rakat directory 'namaz/{rakat_count}' not found"
        )
    
    result = {
        "path": f"data/namaz/{rakat_count}",
        "type": "rakat_count",
        "name": rakat_count,
        "subcategories": get_subdirectories(rakat_path),
        "files": get_files_in_directory(rakat_path),
        "total_files": count_csv_files_in_directory(rakat_path)
    }
    
    return result

@app.get("/data/namaz/{rakat_count}/{subcategory}")
async def get_namaz_subcategory(rakat_count: str, subcategory: str):
    """Get listing of files in a namaz subcategory directory."""
    if rakat_count not in VALID_RAKAT_COUNTS:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid rakat count '{rakat_count}'. Available: {VALID_RAKAT_COUNTS}"
        )
    
    if subcategory not in VALID_NAMAZ_SUBCATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid subcategory '{subcategory}'. Available: {VALID_NAMAZ_SUBCATEGORIES}"
        )
    
    subcategory_path = DATA_DIR / "namaz" / rakat_count / subcategory
    if not subcategory_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Subcategory directory 'namaz/{rakat_count}/{subcategory}' not found"
        )
    
    result = {
        "path": f"data/namaz/{rakat_count}/{subcategory}",
        "type": "subcategory",
        "name": subcategory,
        "rakat_count": rakat_count,
        "files": get_files_in_directory(subcategory_path),
        "total_files": len(get_files_in_directory(subcategory_path))
    }
    
    return result

@app.get("/data/non_namaz/{category}")
async def get_non_namaz_category(category: str):
    """Get listing of files in a non_namaz category directory."""
    if category not in VALID_NON_NAMAZ_CATEGORIES:
        raise HTTPException(
            status_code=404,
            detail=f"Invalid non_namaz category '{category}'. Available: {VALID_NON_NAMAZ_CATEGORIES}"
        )
    
    category_path = DATA_DIR / "non_namaz" / category
    if not category_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Non_namaz category directory 'non_namaz/{category}' not found"
        )
    
    result = {
        "path": f"data/non_namaz/{category}",
        "type": "non_namaz_category",
        "name": category,
        "files": get_files_in_directory(category_path),
        "total_files": len(get_files_in_directory(category_path))
    }
    
    return result



def create_visualization_plots(df, file_path):
    """Create comprehensive visualization plots for 6-axis sensor data."""
    plots = {}
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Calculate magnitudes
    df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # 1. Primary Line Graphs - Individual Axes
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'6-Axis Sensor Data - {file_path.name}', fontsize=16, fontweight='bold')
    
    # Gyroscope data
    axes[0, 0].plot(df['timestamp'], df['gyro_x'], 'b-', linewidth=1.5, label='Gyro X')
    axes[0, 0].set_title('Gyroscope X-Axis')
    axes[0, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(df['timestamp'], df['gyro_y'], 'g-', linewidth=1.5, label='Gyro Y')
    axes[0, 1].set_title('Gyroscope Y-Axis')
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[0, 2].plot(df['timestamp'], df['gyro_z'], 'r-', linewidth=1.5, label='Gyro Z')
    axes[0, 2].set_title('Gyroscope Z-Axis')
    axes[0, 2].set_ylabel('Angular Velocity (rad/s)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Accelerometer data
    axes[1, 0].plot(df['timestamp'], df['acc_x'], 'b-', linewidth=1.5, label='Acc X')
    axes[1, 0].set_title('Accelerometer X-Axis')
    axes[1, 0].set_ylabel('Acceleration (m/s²)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(df['timestamp'], df['acc_y'], 'g-', linewidth=1.5, label='Acc Y')
    axes[1, 1].set_title('Accelerometer Y-Axis')
    axes[1, 1].set_ylabel('Acceleration (m/s²)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    axes[1, 2].plot(df['timestamp'], df['acc_z'], 'r-', linewidth=1.5, label='Acc Z')
    axes[1, 2].set_title('Accelerometer Z-Axis')
    axes[1, 2].set_ylabel('Acceleration (m/s²)')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save line plots
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['line_graphs'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 2. Magnitude Comparison Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle('Sensor Magnitude Analysis', fontsize=14, fontweight='bold')
    
    ax1.plot(df['timestamp'], df['gyro_magnitude'], 'purple', linewidth=2, label='Gyroscope Magnitude')
    ax1.set_title('Gyroscope Magnitude (Total Angular Velocity)')
    ax1.set_ylabel('Magnitude (rad/s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(df['timestamp'], df['acc_magnitude'], 'orange', linewidth=2, label='Accelerometer Magnitude')
    ax2.set_title('Accelerometer Magnitude (Total Acceleration)')
    ax2.set_ylabel('Magnitude (m/s²)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['magnitude_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 3. Scatter Plot - Sensor Relationships
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Sensor Axis Relationships', fontsize=14, fontweight='bold')
    
    # Gyro X vs Y
    scatter1 = axes[0, 0].scatter(df['gyro_x'], df['gyro_y'], c=df['timestamp'], 
                                 cmap='viridis', alpha=0.6, s=1)
    axes[0, 0].set_title('Gyroscope X vs Y')
    axes[0, 0].set_xlabel('Gyro X (rad/s)')
    axes[0, 0].set_ylabel('Gyro Y (rad/s)')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Time (s)')
    
    # Acc X vs Y
    scatter2 = axes[0, 1].scatter(df['acc_x'], df['acc_y'], c=df['timestamp'], 
                                 cmap='plasma', alpha=0.6, s=1)
    axes[0, 1].set_title('Accelerometer X vs Y')
    axes[0, 1].set_xlabel('Acc X (m/s²)')
    axes[0, 1].set_ylabel('Acc Y (m/s²)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Time (s)')
    
    # Gyro magnitude vs Acc magnitude
    scatter3 = axes[1, 0].scatter(df['gyro_magnitude'], df['acc_magnitude'], c=df['timestamp'], 
                                 cmap='coolwarm', alpha=0.6, s=1)
    axes[1, 0].set_title('Gyro Magnitude vs Acc Magnitude')
    axes[1, 0].set_xlabel('Gyro Magnitude (rad/s)')
    axes[1, 0].set_ylabel('Acc Magnitude (m/s²)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Time (s)')
    
    # Gyro Z vs Acc Z (Vertical movement correlation)
    scatter4 = axes[1, 1].scatter(df['gyro_z'], df['acc_z'], c=df['timestamp'], 
                                 cmap='twilight', alpha=0.6, s=1)
    axes[1, 1].set_title('Gyro Z vs Acc Z (Vertical)')
    axes[1, 1].set_xlabel('Gyro Z (rad/s)')
    axes[1, 1].set_ylabel('Acc Z (m/s²)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Time (s)')
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['scatter_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 4. Frequency Analysis (FFT)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Frequency Domain Analysis', fontsize=14, fontweight='bold')
    
    # Calculate sampling rate from timestamps
    if len(df) > 1:
        dt = float(np.mean(np.diff(df['timestamp'].values)))
        fs = 1.0 / dt if dt > 0 else 50  # Default to 50Hz if calculation fails
    else:
        fs = 50
    
    n = len(df)
    freq = fft.fftfreq(n, d=1.0/fs)[:n//2]
    
    # FFT for each gyroscope axis
    gyro_x_data = np.asarray(df['gyro_x'].values, dtype=np.float64)
    gyro_y_data = np.asarray(df['gyro_y'].values, dtype=np.float64)
    gyro_z_data = np.asarray(df['gyro_z'].values, dtype=np.float64)
    
    gyro_x_fft = np.abs(fft.fft(gyro_x_data))[:n//2]
    gyro_y_fft = np.abs(fft.fft(gyro_y_data))[:n//2]
    gyro_z_fft = np.abs(fft.fft(gyro_z_data))[:n//2]
    
    axes[0, 0].semilogy(freq, gyro_x_fft, 'b-', label='Gyro X', alpha=0.8)
    axes[0, 0].semilogy(freq, gyro_y_fft, 'g-', label='Gyro Y', alpha=0.8)
    axes[0, 0].semilogy(freq, gyro_z_fft, 'r-', label='Gyro Z', alpha=0.8)
    axes[0, 0].set_title('Gyroscope Frequency Spectrum')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_xlim(0, 25)  # Focus on 0-25 Hz for prayer movements
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # FFT for each accelerometer axis
    acc_x_data = np.asarray(df['acc_x'].values, dtype=np.float64)
    acc_y_data = np.asarray(df['acc_y'].values, dtype=np.float64)
    acc_z_data = np.asarray(df['acc_z'].values, dtype=np.float64)
    
    acc_x_fft = np.abs(fft.fft(acc_x_data))[:n//2]
    acc_y_fft = np.abs(fft.fft(acc_y_data))[:n//2]
    acc_z_fft = np.abs(fft.fft(acc_z_data))[:n//2]
    
    axes[0, 1].semilogy(freq, acc_x_fft, 'b-', label='Acc X', alpha=0.8)
    axes[0, 1].semilogy(freq, acc_y_fft, 'g-', label='Acc Y', alpha=0.8)
    axes[0, 1].semilogy(freq, acc_z_fft, 'r-', label='Acc Z', alpha=0.8)
    axes[0, 1].set_title('Accelerometer Frequency Spectrum')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_xlim(0, 25)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Magnitude FFT
    gyro_mag_data = np.asarray(df['gyro_magnitude'].values, dtype=np.float64)
    acc_mag_data = np.asarray(df['acc_magnitude'].values, dtype=np.float64)
    
    gyro_mag_fft = np.abs(fft.fft(gyro_mag_data))[:n//2]
    acc_mag_fft = np.abs(fft.fft(acc_mag_data))[:n//2]
    
    axes[1, 0].semilogy(freq, gyro_mag_fft, 'purple', label='Gyro Magnitude', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Gyroscope Magnitude Spectrum')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_xlim(0, 25)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].semilogy(freq, acc_mag_fft, 'orange', label='Acc Magnitude', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('Accelerometer Magnitude Spectrum')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_xlim(0, 25)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['frequency_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 5. Statistical Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Statistical Analysis', fontsize=14, fontweight='bold')
    
    # Data distribution histograms
    gyro_data = [df['gyro_x'], df['gyro_y'], df['gyro_z']]
    gyro_labels = ['Gyro X', 'Gyro Y', 'Gyro Z']
    
    axes[0, 0].hist(gyro_data, bins=30, alpha=0.7, label=gyro_labels, color=['blue', 'green', 'red'])
    axes[0, 0].set_title('Gyroscope Data Distribution')
    axes[0, 0].set_xlabel('Angular Velocity (rad/s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    acc_data = [df['acc_x'], df['acc_y'], df['acc_z']]
    acc_labels = ['Acc X', 'Acc Y', 'Acc Z']
    
    axes[0, 1].hist(acc_data, bins=30, alpha=0.7, label=acc_labels, color=['blue', 'green', 'red'])
    axes[0, 1].set_title('Accelerometer Data Distribution')
    axes[0, 1].set_xlabel('Acceleration (m/s²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots for statistical summary
    gyro_df = df[['gyro_x', 'gyro_y', 'gyro_z']]
    gyro_df.columns = ['Gyro X', 'Gyro Y', 'Gyro Z']
    
    axes[1, 0].boxplot([gyro_df['Gyro X'], gyro_df['Gyro Y'], gyro_df['Gyro Z']], 
                       labels=['Gyro X', 'Gyro Y', 'Gyro Z'])
    axes[1, 0].set_title('Gyroscope Statistical Summary')
    axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 0].grid(True, alpha=0.3)
    
    acc_df = df[['acc_x', 'acc_y', 'acc_z']]
    acc_df.columns = ['Acc X', 'Acc Y', 'Acc Z']
    
    axes[1, 1].boxplot([acc_df['Acc X'], acc_df['Acc Y'], acc_df['Acc Z']], 
                       labels=['Acc X', 'Acc Y', 'Acc Z'])
    axes[1, 1].set_title('Accelerometer Statistical Summary')
    axes[1, 1].set_ylabel('Acceleration (m/s²)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['statistical_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plots, fs

@app.get("/visualizer")
async def get_visualizer():
    """Serve the HTML visualizer interface."""
    html_file = Path(__file__).parent / "visualizer.html"
    if html_file.exists():
        return FileResponse(html_file, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Visualizer not found")

@app.get("/visualize/{path:path}")
async def visualize_csv_data(path: str):
    """
    Visualize 6-axis sensor data from a CSV file.
    
    The path should be relative to the data directory, e.g.:
    /visualize/data/non_namaz/sitting_chair/Siraj_sitting_chair_20260210_073552.csv
    
    Returns comprehensive visualizations including:
    - Line graphs for individual sensor axes
    - Magnitude analysis plots
    - Scatter plots showing sensor relationships
    - Frequency domain analysis (FFT)
    - Statistical distribution analysis
    """
    
    # Security check - ensure path is within data directory
    file_path = DATA_DIR / path
    try:
        file_path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Path must be within the data directory"
        )
    
    # Check if file exists and is a CSV
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}"
        )
    
    if not file_path.suffix == '.csv':
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported for visualization"
        )
    
    try:
        # Read and validate CSV data
        df = pd.read_csv(file_path)
        validate_csv_content(df)
        
        # Create visualizations
        plots, sampling_rate = create_visualization_plots(df, file_path)
        
        # Generate data summary
        gyro_magnitude = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
        acc_magnitude = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        
        # Helper function to safely convert to float
        def safe_float(val):
            return float(val) if val is not None else 0.0
        
        data_summary = {
            "file_info": {
                "filename": file_path.name,
                "path": str(file_path.relative_to(DATA_DIR)),
                "size_bytes": file_path.stat().st_size,
                "rows": len(df),
                "duration_seconds": safe_float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]),
                "sampling_rate_hz": round(sampling_rate, 2)
            },
            "sensor_summary": {
                "gyroscope": {
                    "x_range": [safe_float(df['gyro_x'].min()), safe_float(df['gyro_x'].max())],
                    "y_range": [safe_float(df['gyro_y'].min()), safe_float(df['gyro_y'].max())],
                    "z_range": [safe_float(df['gyro_z'].min()), safe_float(df['gyro_z'].max())],
                    "magnitude_range": [safe_float(gyro_magnitude.min()), safe_float(gyro_magnitude.max())],
                    "x_mean": safe_float(df['gyro_x'].mean()),
                    "y_mean": safe_float(df['gyro_y'].mean()),
                    "z_mean": safe_float(df['gyro_z'].mean()),
                    "magnitude_mean": safe_float(gyro_magnitude.mean())
                },
                "accelerometer": {
                    "x_range": [safe_float(df['acc_x'].min()), safe_float(df['acc_x'].max())],
                    "y_range": [safe_float(df['acc_y'].min()), safe_float(df['acc_y'].max())],
                    "z_range": [safe_float(df['acc_z'].min()), safe_float(df['acc_z'].max())],
                    "magnitude_range": [safe_float(acc_magnitude.min()), safe_float(acc_magnitude.max())],
                    "x_mean": safe_float(df['acc_x'].mean()),
                    "y_mean": safe_float(df['acc_y'].mean()),
                    "z_mean": safe_float(df['acc_z'].mean()),
                    "magnitude_mean": safe_float(acc_magnitude.mean())
                }
            }
        }
        
        return {
            "success": True,
            "message": f"Visualization generated for {file_path.name}",
            "data_summary": data_summary,
            "visualizations": {
                "line_graphs": {
                    "title": "Individual Sensor Axes - Time Series",
                    "description": "Line graphs showing each sensor axis over time",
                    "image_data": plots['line_graphs']
                },
                "magnitude_analysis": {
                    "title": "Sensor Magnitude Analysis",
                    "description": "Combined magnitude plots showing total angular velocity and acceleration",
                    "image_data": plots['magnitude_analysis']
                },
                "scatter_analysis": {
                    "title": "Sensor Axis Relationships",
                    "description": "Scatter plots showing correlations between different sensor axes",
                    "image_data": plots['scatter_analysis']
                },
                "frequency_analysis": {
                    "title": "Frequency Domain Analysis (FFT)",
                    "description": "Fast Fourier Transform analysis showing frequency components (0-25 Hz focus)",
                    "image_data": plots['frequency_analysis']
                },
                "statistical_analysis": {
                    "title": "Statistical Distribution Analysis",
                    "description": "Histograms and box plots showing data distribution and statistical summary",
                    "image_data": plots['statistical_analysis']
                }
            },
            "usage_notes": {
                "interpretation": {
                    "line_graphs": "Look for repeating patterns indicating prayer movements (ruku, sujud, etc.)",
                    "magnitude_analysis": "Peaks indicate significant movements, useful for counting prayer cycles",
                    "scatter_analysis": "Clusters may indicate different prayer positions or movement phases",
                    "frequency_analysis": "Dominant frequencies help characterize movement patterns and distinguish from noise",
                    "statistical_analysis": "Distribution shape and outliers can indicate data quality and movement characteristics"
                },
                "prayer_detection_tips": {
                    "ruku_patterns": "Look for distinctive bowing patterns in gyroscope Y-axis (forward tilt)",
                    "sujud_patterns": "Prostration shows characteristic patterns in accelerometer Z-axis (gravity changes)",
                    "timing_analysis": "Use timestamp data to analyze prayer duration and rakat timing",
                    "movement_regularity": "Consistent patterns indicate structured prayer movements vs random activity"
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)