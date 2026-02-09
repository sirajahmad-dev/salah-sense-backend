# Salahsense API Documentation

## Overview
The Salahsense API provides RESTful endpoints for managing training data, training models, and making predictions for the 6-axis namaz detection system.

**Base URL**: `http://localhost:8000`

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python3 api.py
```
The API will be available at `http://localhost:8000`

### 3. View Interactive Docs
Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## API Endpoints

### 1. Upload Training Data
**POST** `/upload`

Upload a CSV file to the appropriate data category.

**Parameters:**
- `file` (multipart/form-data): CSV file to upload
- `category_type` (form): `"namaz"` or `"non_namaz"`
- `category` (form): Specific subcategory name

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_data.csv" \
  -F "category_type=namaz" \
  -F "category=standing_loose_clothing"
```

**Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "file_info": {
    "filename": "20240130_143052_a1b2c3d4_your_data.csv",
    "category_type": "namaz",
    "category": "standing_loose_clothing",
    "rows": 3000,
    "columns": 7
  },
  "data_statistics": {
    "namaz": {
      "standing_loose_clothing": 25,
      "standing_tight_clothing": 18,
      "sitting_floor": 12
    },
    "non_namaz": {
      "walking": 30,
      "running": 25,
      "sitting_floor": 15,
      "sitting_chair": 20
    }
  }
}
```

### 2. Get Available Categories
**GET** `/categories`

Returns list of valid category names.

**Response:**
```json
{
  "namaz": [
    "standing_loose_clothing",
    "standing_tight_clothing", 
    "sitting_floor"
  ],
  "non_namaz": [
    "walking",
    "running",
    "sitting_floor",
    "sitting_chair"
  ]
}
```

### 3. Get System Status
**GET** `/status`

Get current data statistics and model status.

**Response:**
```json
{
  "status": "ready",
  "model_trained": false,
  "data_statistics": {
    "total_files": 145,
    "total_namaz": 55,
    "total_non_namaz": 90,
    "namaz": {
      "standing_loose_clothing": 25,
      "standing_tight_clothing": 18,
      "sitting_floor": 12
    },
    "non_namaz": {
      "walking": 30,
      "running": 25,
      "sitting_floor": 15,
      "sitting_chair": 20
    }
  },
  "ready_for_training": true,
  "recommendation": {
    "min_files_per_category": 50,
    "optimal_files_per_category": 200,
    "total_optimal_files": 1400
  }
}
```

### 4. Train Model
**POST** `/train`

Start model training on uploaded data. This is a background task.

**Response:**
```json
{
  "success": true,
  "message": "Training started in background",
  "job_id": "f4e8d9a1",
  "log_file": "uploads/training_f4e8d9a1.log",
  "current_data_stats": {...},
  "check_status_endpoint": "/train_status/f4e8d9a1"
}
```

### 5. Check Training Status
**GET** `/train_status/{job_id}`

Monitor the progress of a training job.

**Response:**
```json
{
  "job_id": "f4e8d9a1",
  "status": "completed",
  "log": "2024-01-30 14:30:00: STEP 1: Loading Data...",
  "model_exists": true
}
```

### 6. Make Prediction
**POST** `/predict`

Make a prediction on a new CSV file.

**Parameters:**
- `file` (multipart/form-data): CSV file for prediction

**Response:**
```json
{
  "success": true,
  "prediction": "NAMAZ",
  "confidence": 92.5,
  "probabilities": {
    "namaz": 92.5,
    "non_namaz": 7.5
  },
  "file_info": {
    "filename": "test.csv",
    "rows": 2000,
    "processed_at": "2024-01-30T14:35:00"
  }
}
```

### 7. Delete Data File
**DELETE** `/data/{category_type}/{category}/{filename}`

Delete a specific training data file.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/data/namaz/standing_loose_clothing/20240130_143052_file.csv"
```

### 8. Data Listing Endpoints
**GET** `/data`

Get listing of all data in root data folder with file counts.

**Response:**
```json
{
  "path": "data",
  "type": "root",
  "categories": {
    "namaz": {
      "type": "category",
      "subcategories": [
        {
          "name": "2_rakat",
          "file_count": 25,
          "path": "namaz/2_rakat"
        },
        {
          "name": "3_rakat", 
          "file_count": 18,
          "path": "namaz/3_rakat"
        },
        {
          "name": "4_rakat",
          "file_count": 12,
          "path": "namaz/4_rakat"
        }
      ],
      "total_files": 55
    },
    "non_namaz": {
      "type": "category",
      "subcategories": [
        {
          "name": "walking",
          "file_count": 30,
          "path": "non_namaz/walking"
        },
        {
          "name": "running",
          "file_count": 25,
          "path": "non_namaz/running"
        }
      ],
      "total_files": 55
    }
  },
  "total_files": 110
}
```

---

**GET** `/data/{category}`

Get listing of files in a specific category (namaz or non_namaz).

**Examples:**
- `GET /data/namaz` - List all namaz rakat counts
- `GET /data/non_namaz` - List all non_namaz activities

**Response:**
```json
{
  "path": "data/namaz",
  "type": "category",
  "name": "namaz",
  "subcategories": [
    {
      "name": "2_rakat",
      "file_count": 25,
      "path": "namaz/2_rakat"
    }
  ],
  "files": [],
  "total_files": 25
}
```

---

**GET** `/data/namaz/{rakat_count}`

Get listing of files in a specific namaz rakat count directory.

**Examples:**
- `GET /data/namaz/2_rakat` - List 2 rakat namaz subcategories
- `GET /data/namaz/3_rakat` - List 3 rakat namaz subcategories
- `GET /data/namaz/4_rakat` - List 4 rakat namaz subcategories

**Response:**
```json
{
  "path": "data/namaz/2_rakat",
  "type": "rakat_count",
  "name": "2_rakat",
  "subcategories": [
    {
      "name": "standing_loose_clothing",
      "file_count": 15,
      "path": "namaz/2_rakat/standing_loose_clothing"
    },
    {
      "name": "standing_tight_clothing",
      "file_count": 10,
      "path": "namaz/2_rakat/standing_tight_clothing"
    }
  ],
  "files": [],
  "total_files": 25
}
```

---

**GET** `/data/namaz/{rakat_count}/{subcategory}`

Get listing of CSV files in a specific namaz subcategory.

**Examples:**
- `GET /data/namaz/2_rakat/standing_loose_clothing` - List files in 2 rakat standing loose clothing
- `GET /data/namaz/3_rakat/sitting_floor` - List files in 3 rakat sitting floor

**Response:**
```json
{
  "path": "data/namaz/2_rakat/standing_loose_clothing",
  "type": "subcategory",
  "name": "standing_loose_clothing",
  "rakat_count": "2_rakat",
  "files": [
    {
      "name": "user1_2_rakat_standing_loose_clothing_20240130_143052.csv",
      "size": 15420,
      "modified": "2024-01-30T14:30:52.123456",
      "path": "namaz/2_rakat/standing_loose_clothing/user1_2_rakat_standing_loose_clothing_20240130_143052.csv"
    }
  ],
  "total_files": 15
}
```

---

**GET** `/data/non_namaz/{category}`

Get listing of CSV files in a specific non_namaz activity category.

**Examples:**
- `GET /data/non_namaz/walking` - List walking activity files
- `GET /data/non_namaz/running` - List running activity files
- `GET /data/non_namaz/sitting_floor` - List sitting floor activity files
- `GET /data/non_namaz/sitting_chair` - List sitting chair activity files

**Response:**
```json
{
  "path": "data/non_namaz/walking",
  "type": "non_namaz_category",
  "name": "walking",
  "files": [
    {
      "name": "user1_walking_20240130_143052.csv",
      "size": 12450,
      "modified": "2024-01-30T14:30:52.123456",
      "path": "non_namaz/walking/user1_walking_20240130_143052.csv"
    }
  ],
  "total_files": 30
}
```

### 9. Download Model
**GET** `/download/model`

Download the trained model file.

## CSV File Format

**Required Columns (7 total):**
```
timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
```

**Example Data:**
```csv
timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
0.00, -0.177, 0.216, -0.137, 0.123, -0.456, 9.807
0.01, -0.244, -0.468, -0.277, 0.145, -0.432, 9.812
...
```

## Mobile App Integration

### Upload Flow
1. **Check Categories**: `GET /categories` to get valid categories
2. **Upload File**: `POST /upload` with appropriate `category_type` and `category`
3. **Monitor Upload**: Check response for updated statistics
4. **Train Model**: `POST /train` when enough data is collected
5. **Monitor Training**: `GET /train_status/{job_id}` for progress

### Prediction Flow
1. **Check Model**: `GET /status` to verify `model_trained: true`
2. **Upload File**: `POST /predict` with new CSV file
3. **Get Result**: Classification with confidence percentages

### Example Mobile App Integration

```javascript
// Upload training data
async function uploadTrainingData(csvFile, categoryType, category) {
  const formData = new FormData();
  formData.append('file', csvFile);
  formData.append('category_type', categoryType);
  formData.append('category', category);
  
  const response = await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Make prediction
async function makePrediction(csvFile) {
  const formData = new FormData();
  formData.append('file', csvFile);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}
```

## Data Categories

### Namaz Categories
- `standing_loose_clothing`: Prayer with loose clothing (kurta, thawb)
- `standing_tight_clothing`: Prayer with tight clothing (shirt, jeans)
- `sitting_floor`: Floor-based sitting prayer

### Non-Namaz Categories
- `walking`: Walking activities
- `running`: Running activities
- `sitting_floor`: Sitting on floor (not praying)
- `sitting_chair`: Sitting on chair

## Error Handling

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid data format, missing parameters)
- `404`: Not Found (file or job not found)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message describing the issue"
}
```

## Security Notes

- API accepts connections from any origin (configured for mobile app)
- File uploads are validated for content and format
- Temporary files are automatically cleaned up
- Training runs in background with logging

## Performance Considerations

- **Training**: Background task, can take several minutes
- **Prediction**: Usually completes within seconds
- **File Upload**: Limited by server configuration
- **Data Storage**: Ensure sufficient disk space for training data

## Troubleshooting

### "No training data available"
- Upload CSV files first using `/upload` endpoint
- Check `/status` to see current data statistics

### "Model not trained"
- Run `/train` endpoint first
- Check training status using `/train_status/{job_id}`

### "Invalid CSV format"
- Ensure CSV has exactly 7 required columns
- Check that all numeric columns contain valid numbers
- Minimum 50 rows of data required