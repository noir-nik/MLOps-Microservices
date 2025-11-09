from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import yaml
from pathlib import Path
from datetime import datetime
import logging
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import requests
import asyncio
from queue import Queue
import threading

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
        logging.FileHandler(log_dir / "ml_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Service", version="1.0.0")

# Models directory
models_dir = Path(config["storage"]["models_dir"])
models_dir.mkdir(parents=True, exist_ok=True)

# Progress tracking
progress_queues: Dict[int, Queue] = {}

class TrainRequest(BaseModel):
    dataset_id: int
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = {}
    run_id: int

class InferRequest(BaseModel):
    model_id: int
    data: str  # CSV string
    run_id: int

def preprocess_titanic(df):
    """Preprocess Titanic dataset"""
    df = df.copy()
    
    # Drop unnecessary columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Fill missing values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Encode categorical variables
    le = LabelEncoder()
    if 'Sex' in df.columns:
        df['Sex'] = le.fit_transform(df['Sex'])
    if 'Embarked' in df.columns:
        df['Embarked'] = le.fit_transform(df['Embarked'])
    
    return df

def send_progress(run_id: int, progress: float, message: str):
    """Send progress update to storage"""
    try:
        if run_id in progress_queues:
            progress_queues[run_id].put({"progress": progress, "message": message})
        logger.info(f"Run {run_id}: {progress:.0f}% - {message}")
    except Exception as e:
        logger.error(f"Error sending progress: {e}")

def update_run_status(run_id: int, status: str, result: Optional[Dict] = None):
    """Update run status in storage"""
    try:
        storage_url = config["services"]["storage_url"]
        data = {"status": status}
        if result:
            data["result"] = result
        if status == "completed":
            data["completed_at"] = datetime.now().isoformat()
        
        response = requests.patch(f"{storage_url}/runs/{run_id}", json=data)
        response.raise_for_status()
        logger.info(f"Run {run_id} status updated to {status}")
    except Exception as e:
        logger.error(f"Error updating run status: {e}")

def train_model_task(dataset_id: int, model_type: str, hyperparameters: Dict, run_id: int):
    """Background task for model training"""
    try:
        send_progress(run_id, 0, "Starting training")
        update_run_status(run_id, "running")
        
        # Get dataset from storage
        storage_url = config["services"]["storage_url"]
        send_progress(run_id, 10, "Loading dataset")
        
        dataset_response = requests.get(f"{storage_url}/datasets/{dataset_id}")
        dataset_response.raise_for_status()
        dataset_info = dataset_response.json()
        
        # Load data
        data_path = Path(dataset_info["file_path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        send_progress(run_id, 20, f"Loaded {len(df)} rows")
        
        # Preprocess
        send_progress(run_id, 30, "Preprocessing data")
        df = preprocess_titanic(df)
        
        # Split features and target
        if 'Survived' not in df.columns:
            raise ValueError("Target column 'Survived' not found")
        
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        send_progress(run_id, 40, f"Split data: {len(X_train)} train, {len(X_test)} test")
        
        # Create model
        send_progress(run_id, 50, f"Creating {model_type} model")
        if model_type == "LogisticRegression":
            model = LogisticRegression(
                max_iter=hyperparameters.get("max_iter", 1000),
                C=hyperparameters.get("C", 1.0),
                random_state=42
            )
        elif model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=hyperparameters.get("n_estimators", 100),
                max_depth=hyperparameters.get("max_depth", None),
                random_state=42
            )
        elif model_type == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=hyperparameters.get("n_neighbors", 5),
                weights=hyperparameters.get("weights", "uniform")
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        send_progress(run_id, 60, "Training model")
        model.fit(X_train, y_train)
        
        # Evaluate
        send_progress(run_id, 80, "Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_[0]).tolist()))
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Save model
        send_progress(run_id, 90, "Saving model")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"
        model_path = models_dir / f"{model_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': list(X.columns),
                'model_type': model_type,
                'hyperparameters': hyperparameters
            }, f)
        
        # Register model in storage
        model_data = {
            "name": model_name,
            "model_type": model_type,
            "version": "1.0",
            "file_path": str(model_path),
            "dataset_id": dataset_id,
            "accuracy": float(accuracy),
            "meta": {
                "hyperparameters": hyperparameters,
                "feature_importance": feature_importance,
                "classification_report": report,
                "confusion_matrix": conf_matrix,
                "train_size": len(X_train),
                "test_size": len(X_test)
            }
        }
        
        model_response = requests.post(f"{storage_url}/models", json=model_data)
        model_response.raise_for_status()
        model_id = model_response.json()["id"]
        
        send_progress(run_id, 100, "Training completed")
        
        # Update run with results
        result = {
            "model_id": model_id,
            "accuracy": float(accuracy),
            "model_name": model_name,
            "feature_importance": feature_importance
        }
        update_run_status(run_id, "completed", result)
        
        logger.info(f"Training completed: {model_name}, accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        update_run_status(run_id, "failed", {"error": str(e)})
        send_progress(run_id, 100, f"Training failed: {str(e)}")

def inference_task(model_id: int, data_csv: str, run_id: int):
    """Background task for model inference"""
    try:
        send_progress(run_id, 0, "Starting inference")
        update_run_status(run_id, "running")
        
        storage_url = config["services"]["storage_url"]
        
        # Get model from storage
        send_progress(run_id, 20, "Loading model")
        model_response = requests.get(f"{storage_url}/models/{model_id}")
        model_response.raise_for_status()
        model_info = model_response.json()
        
        # Load model file
        model_path = Path(model_info["file_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Parse input data
        send_progress(run_id, 40, "Preprocessing data")
        from io import StringIO
        df = pd.read_csv(StringIO(data_csv))
        
        # Preprocess
        df = preprocess_titanic(df)
        
        # Ensure features match
        for col in feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing feature: {col}")
        
        X = df[feature_names]
        
        # Predict
        send_progress(run_id, 70, "Running predictions")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Save results
        send_progress(run_id, 90, "Saving results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(config["storage"]["results_dir"]) / f"inference_{timestamp}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df = df.copy()
        results_df['Prediction'] = predictions
        if probabilities is not None:
            results_df['Probability_0'] = probabilities[:, 0]
            results_df['Probability_1'] = probabilities[:, 1]
        
        results_df.to_csv(results_path, index=False)
        
        send_progress(run_id, 100, "Inference completed")
        
        # Update run
        result = {
            "predictions": predictions.tolist(),
            "results_file": str(results_path),
            "num_predictions": len(predictions)
        }
        update_run_status(run_id, "completed", result)
        
        logger.info(f"Inference completed: {len(predictions)} predictions")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        update_run_status(run_id, "failed", {"error": str(e)})
        send_progress(run_id, 100, f"Inference failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("ML Service started")

@app.get("/")
async def root():
    return {
        "service": "ML Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    try:
        # Create progress queue
        progress_queues[request.run_id] = Queue()
        
        # Start training in background
        background_tasks.add_task(
            train_model_task,
            request.dataset_id,
            request.model_type,
            request.hyperparameters,
            request.run_id
        )
        
        logger.info(f"Training started for run {request.run_id}")
        return {"message": "Training started", "run_id": request.run_id}
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer_model(request: InferRequest, background_tasks: BackgroundTasks):
    """Start model inference"""
    try:
        # Create progress queue
        progress_queues[request.run_id] = Queue()
        
        # Start inference in background
        background_tasks.add_task(
            inference_task,
            request.model_id,
            request.data,
            request.run_id
        )
        
        logger.info(f"Inference started for run {request.run_id}")
        return {"message": "Inference started", "run_id": request.run_id}
    except Exception as e:
        logger.error(f"Error starting inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{run_id}")
async def get_progress(run_id: int):
    """SSE endpoint for progress updates"""
    async def event_generator():
        queue = progress_queues.get(run_id)
        if not queue:
            yield f"data: {json.dumps({'error': 'Run not found'})}\n\n"
            return
        
        while True:
            try:
                if not queue.empty():
                    progress_data = queue.get_nowait()
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    if progress_data.get("progress", 0) >= 100:
                        break
                else:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in progress stream: {e}")
                break
        
        # Cleanup
        if run_id in progress_queues:
            del progress_queues[run_id]
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["service"]["port"])
