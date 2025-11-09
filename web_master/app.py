from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any
import yaml
from pathlib import Path
from datetime import datetime
import logging
import requests
import json
import asyncio

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
        logging.FileHandler(log_dir / "web_master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Web Master", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
COLLECTOR_URL = config["services"]["collector_url"]
STORAGE_URL = config["services"]["storage_url"]
ML_SERVICE_URL = config["services"]["ml_service_url"]

class CollectRequest(BaseModel):
    batch_size: int = 100

class TrainRequest(BaseModel):
    dataset_id: int
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = {}

class InferRequest(BaseModel):
    model_id: int
    data: str  # CSV string

@app.on_event("startup")
async def startup_event():
    logger.info("Web Master started")

@app.get("/")
async def root():
    return {
        "service": "Web Master",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "collector": COLLECTOR_URL,
            "storage": STORAGE_URL,
            "ml_service": ML_SERVICE_URL
        }
    }

@app.get("/health")
async def health():
    """Health check for all services"""
    health_status = {}
    
    try:
        response = requests.get(f"{COLLECTOR_URL}/health", timeout=2)
        health_status["collector"] = "healthy" if response.ok else "unhealthy"
    except:
        health_status["collector"] = "unreachable"
    
    try:
        response = requests.get(f"{STORAGE_URL}/health", timeout=2)
        health_status["storage"] = "healthy" if response.ok else "unhealthy"
    except:
        health_status["storage"] = "unreachable"
    
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=2)
        health_status["ml_service"] = "healthy" if response.ok else "unhealthy"
    except:
        health_status["ml_service"] = "unreachable"
    
    all_healthy = all(status == "healthy" for status in health_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": health_status
    }

# Scenario 1: Collect and Store
@app.post("/scenarios/collect-and-store")
async def collect_and_store(request: CollectRequest):
    """
    Scenario 1: Collect batch from collector and store in storage
    """
    try:
        logger.info(f"Starting collect-and-store: batch_size={request.batch_size}")
        
        # Get batch from collector
        response = requests.get(
            f"{COLLECTOR_URL}/batch",
            params={"size": request.batch_size}
        )
        response.raise_for_status()
        
        csv_data = response.text
        batch_start = response.headers.get("X-Batch-Start", "0")
        batch_end = response.headers.get("X-Batch-End", str(request.batch_size))
        
        # Save to storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"titanic_batch_{batch_start}_{batch_end}_{timestamp}"
        
        # Upload file
        files = {"file": (f"{dataset_name}.csv", csv_data, "text/csv")}
        upload_response = requests.post(f"{STORAGE_URL}/upload/dataset", files=files)
        upload_response.raise_for_status()
        upload_data = upload_response.json()
        
        # Count rows and columns
        import pandas as pd
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
        
        # Create dataset record
        dataset_data = {
            "name": dataset_name,
            "file_path": upload_data["path"],
            "rows": len(df),
            "columns": len(df.columns),
            "meta": {
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "source": "collector"
            }
        }
        
        dataset_response = requests.post(f"{STORAGE_URL}/datasets", json=dataset_data)
        dataset_response.raise_for_status()
        dataset_id = dataset_response.json()["id"]
        
        logger.info(f"Dataset created: {dataset_name} (ID: {dataset_id})")
        
        return {
            "message": "Batch collected and stored successfully",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Collect-and-store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario 2: Train Model
@app.post("/scenarios/train-model")
async def train_model(request: TrainRequest):
    """
    Scenario 2: Train a model with selected dataset and model type
    """
    try:
        logger.info(f"Starting training: dataset={request.dataset_id}, model={request.model_type}")
        
        # Verify dataset exists
        dataset_response = requests.get(f"{STORAGE_URL}/datasets/{request.dataset_id}")
        dataset_response.raise_for_status()
        
        # Create run record
        run_data = {
            "run_type": "training",
            "dataset_id": request.dataset_id,
            "status": "pending",
            "meta": {
                "model_type": request.model_type,
                "hyperparameters": request.hyperparameters
            }
        }
        
        run_response = requests.post(f"{STORAGE_URL}/runs", json=run_data)
        run_response.raise_for_status()
        run_id = run_response.json()["id"]
        
        # Start training
        train_request_data = {
            "dataset_id": request.dataset_id,
            "model_type": request.model_type,
            "hyperparameters": request.hyperparameters,
            "run_id": run_id
        }
        
        train_response = requests.post(f"{ML_SERVICE_URL}/train", json=train_request_data)
        train_response.raise_for_status()
        
        logger.info(f"Training started: run_id={run_id}")
        
        return {
            "message": "Training started",
            "run_id": run_id,
            "model_type": request.model_type
        }
        
    except Exception as e:
        logger.error(f"Train model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario 3: Infer Model
@app.post("/scenarios/infer-model")
async def infer_model(request: InferRequest):
    """
    Scenario 3: Run inference with a trained model
    """
    try:
        logger.info(f"Starting inference: model_id={request.model_id}")
        
        # Verify model exists
        model_response = requests.get(f"{STORAGE_URL}/models/{request.model_id}")
        model_response.raise_for_status()
        model_info = model_response.json()
        
        # Create run record
        run_data = {
            "run_type": "inference",
            "model_id": request.model_id,
            "status": "pending",
            "meta": {
                "model_name": model_info["name"]
            }
        }
        
        run_response = requests.post(f"{STORAGE_URL}/runs", json=run_data)
        run_response.raise_for_status()
        run_id = run_response.json()["id"]
        
        # Start inference
        infer_request_data = {
            "model_id": request.model_id,
            "data": request.data,
            "run_id": run_id
        }
        
        infer_response = requests.post(f"{ML_SERVICE_URL}/infer", json=infer_request_data)
        infer_response.raise_for_status()
        
        logger.info(f"Inference started: run_id={run_id}")
        
        return {
            "message": "Inference started",
            "run_id": run_id,
            "model_name": model_info["name"]
        }
        
    except Exception as e:
        logger.error(f"Infer model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scenario 4: Get Report
@app.get("/scenarios/report/{model_id}")
async def get_report(model_id: int):
    """
    Scenario 4: Get detailed report for a trained model
    """
    try:
        logger.info(f"Generating report for model {model_id}")
        
        # Get model info
        model_response = requests.get(f"{STORAGE_URL}/models/{model_id}")
        model_response.raise_for_status()
        model_info = model_response.json()
        
        # Get dataset info
        dataset_response = requests.get(f"{STORAGE_URL}/datasets/{model_info['dataset_id']}")
        dataset_response.raise_for_status()
        dataset_info = dataset_response.json()
        
        # Parse meta
        meta = model_info.get("meta", {})
        
        report = {
            "model": {
                "id": model_info["id"],
                "name": model_info["name"],
                "type": model_info["model_type"],
                "version": model_info["version"],
                "created_at": model_info["created_at"]
            },
            "dataset": {
                "id": dataset_info["id"],
                "name": dataset_info["name"],
                "rows": dataset_info["rows"],
                "columns": dataset_info["columns"]
            },
            "performance": {
                "accuracy": model_info.get("accuracy"),
                "train_size": meta.get("train_size"),
                "test_size": meta.get("test_size")
            },
            "hyperparameters": meta.get("hyperparameters", {}),
            "feature_importance": meta.get("feature_importance", {}),
            "classification_report": meta.get("classification_report", {}),
            "confusion_matrix": meta.get("confusion_matrix", [])
        }
        
        logger.info(f"Report generated for model {model_id}")
        
        return report
        
    except Exception as e:
        logger.error(f"Get report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Progress streaming endpoint
@app.get("/progress/{run_id}")
async def stream_progress(run_id: int):
    """Stream progress updates from ML service"""
    try:
        async def event_generator():
            async with requests.get(
                f"{ML_SERVICE_URL}/progress/{run_id}",
                stream=True,
                timeout=300
            ) as response:
                for line in response.iter_lines():
                    if line:
                        yield line.decode('utf-8') + '\n'
                        await asyncio.sleep(0)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Progress streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Proxy endpoints for direct access to storage
@app.get("/datasets")
async def list_datasets():
    response = requests.get(f"{STORAGE_URL}/datasets")
    return response.json()

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int):
    response = requests.get(f"{STORAGE_URL}/datasets/{dataset_id}")
    return response.json()

@app.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int):
    response = requests.get(f"{STORAGE_URL}/datasets/{dataset_id}/download")
    # print(response.headers)
    return StreamingResponse(response.iter_content(chunk_size=1024), headers=response.headers, status_code=response.status_code, media_type="text/csv")


@app.get("/models")
async def list_models():
    response = requests.get(f"{STORAGE_URL}/models")
    return response.json()

@app.get("/models/{model_id}")
async def get_model(model_id: int):
    response = requests.get(f"{STORAGE_URL}/models/{model_id}")
    return response.json()

@app.get("/models/{model_id}/download")
async def download_model(model_id: int):
    response = requests.get(f"{STORAGE_URL}/models/{model_id}/download")
    return StreamingResponse(response.iter_content(chunk_size=1024), headers=response.headers, status_code=response.status_code, media_type="application/octet-stream")

@app.get("/runs")
async def list_runs():
    response = requests.get(f"{STORAGE_URL}/runs")
    return response.json()

@app.get("/runs/{run_id}")
async def get_run(run_id: int):
    response = requests.get(f"{STORAGE_URL}/runs/{run_id}")
    return response.json()

@app.get("/files")
async def list_files():
    response = requests.get(f"{STORAGE_URL}/files")
    return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["service"]["port"])
