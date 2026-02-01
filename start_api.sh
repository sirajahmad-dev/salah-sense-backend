#!/bin/bash
# API Server Startup Script for Salahsense

echo "=========================================="
echo "Salahsense Rakat Detection API Server - Startup"
echo "=========================================="
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, pandas, numpy, sklearn, scipy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed"
else
    echo "✗ Missing dependencies. Installing..."
    pip3 install -r requirements.txt
fi
echo ""

# Create directories
echo "Setting up directories..."
python3 -c "
from pathlib import Path
Path('data').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True) 
Path('uploads').mkdir(exist_ok=True)
print('✓ Directories ready')
"
echo ""

# Start API server
echo "Starting Salahsense API Server..."
echo "API will be available at: http://localhost:8000"
echo "Interactive docs: http://localhost:8000/docs"
echo "Rakat API documentation: RAKAT_API_DOCUMENTATION.md"
echo "Basic API documentation: API_DOCUMENTATION.md"
echo ""
echo "New: Upload with rakat_count parameter!"
echo "Example: curl -X POST http://localhost:8000/upload -F file=@data.csv -F category_type=namaz -F category=standing_loose_clothing -F rakat_count=3_rakat"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo ""

python3 api.py