from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import logging
import io

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup logging
log_dir = Path(config["logging"]["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Collector Service", version="1.0.0")

# Load dataset
data_path = Path(config["data"]["source_path"])
df_full = None
current_offset = 0

def load_dataset():
    global df_full
    try:
        df_full = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully: {len(df_full)} rows")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

load_dataset()

@app.on_event("startup")
async def startup_event():
    logger.info("Collector service started")

@app.get("/")
async def root():
    return {
        "service": "Collector",
        "version": "1.0.0",
        "status": "running",
        "total_rows": len(df_full) if df_full is not None else 0
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/batch")
async def get_batch(size: int = 100, offset: int = 0):
    """
    Get a batch of data from the dataset.
    
    Args:
        size: Number of rows to return
        offset: Starting position in the dataset
    
    Returns:
        CSV data as text
    """
    global current_offset
    
    if df_full is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")
    
    if size <= 0:
        raise HTTPException(status_code=400, detail="Size must be positive")
    
    # Use provided offset or continue from last position
    if current_offset + size > len(df_full) and offset == 0:
        current_offset = 0
    start = offset if offset > 0 else current_offset
    end = min(start + size, len(df_full))
    
    if start >= len(df_full):
        logger.warning(f"Offset {start} exceeds dataset size {len(df_full)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Offset {start} exceeds dataset size {len(df_full)}"
        )
    
    batch = df_full.iloc[start:end]
    current_offset = end
    
    # Log the request
    logger.info(f"Batch requested: size={size}, offset={start}, returned={len(batch)} rows")
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    batch.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    return StreamingResponse(
        io.StringIO(csv_data),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=batch_{start}_{end}.csv",
            "X-Batch-Start": str(start),
            "X-Batch-End": str(end),
            "X-Batch-Size": str(len(batch))
        }
    )

@app.get("/info")
async def get_info():
    """Get information about the dataset"""
    if df_full is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    
    return {
        "total_rows": len(df_full),
        "columns": list(df_full.columns),
        "current_offset": current_offset,
        "remaining_rows": len(df_full) - current_offset
    }

@app.post("/reset")
async def reset_offset():
    """Reset the current offset to 0"""
    global current_offset
    current_offset = 0
    logger.info("Offset reset to 0")
    return {"message": "Offset reset", "current_offset": current_offset}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["service"]["port"])
