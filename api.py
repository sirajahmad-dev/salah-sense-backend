"""
FastAPI interface for Salahsense Namaz Detection System.

This API provides endpoints for:
1. Uploading CSV training data to appropriate subfolders
2. Training the model on uploaded data
3. Making predictions on new CSV files
4. Getting system status and statistics
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import json
import subprocess
from typing import Optional
import asyncio
import io


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
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
UPLOAD_DIR = Path("uploads")

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
            "categories": "/categories"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)