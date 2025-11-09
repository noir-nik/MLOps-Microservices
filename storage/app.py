from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from pydantic import BaseModel
from typing import Optional, List
import yaml
from pathlib import Path
from datetime import datetime
import logging
import json
import shutil

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
        logging.FileHandler(log_dir / "storage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup database
db_path = Path(config["database"]["path"])
db_path.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(config["database"]["url"])
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Setup data directories
data_dir = Path(config["storage"]["data_dir"])
datasets_dir = data_dir / "datasets"
models_dir = data_dir / "models"
results_dir = data_dir / "results"

for dir_path in [datasets_dir, models_dir, results_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database Models
class FileMetadata(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    path = Column(String)
    size = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(Text)

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    file_path = Column(String)
    rows = Column(Integer)
    columns = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(Text)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    model_type = Column(String)
    version = Column(String)
    file_path = Column(String)
    dataset_id = Column(Integer)
    accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    meta = Column(Text)

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    run_type = Column(String)  # 'training' or 'inference'
    model_id = Column(Integer, nullable=True)
    dataset_id = Column(Integer, nullable=True)
    status = Column(String)  # 'pending', 'running', 'completed', 'failed'
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    meta = Column(Text)

Base.metadata.create_all(bind=engine)

# Pydantic models
class FileCreate(BaseModel):
    name: str
    path: str
    size: int
    meta: Optional[dict] = {}

class DatasetCreate(BaseModel):
    name: str
    file_path: str
    rows: int
    columns: int
    meta: Optional[dict] = {}

class ModelCreate(BaseModel):
    name: str
    model_type: str
    version: str
    file_path: str
    dataset_id: int
    accuracy: Optional[float] = None
    meta: Optional[dict] = {}

class RunCreate(BaseModel):
    run_type: str
    model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    status: str = "pending"
    meta: Optional[dict] = {}

class RunUpdate(BaseModel):
    status: Optional[str] = None
    result: Optional[dict] = None
    completed_at: Optional[datetime] = None

app = FastAPI(title="Storage Service", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    logger.info("Storage service started")

@app.get("/")
async def root():
    return {
        "service": "Storage",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Files endpoints
@app.post("/files")
async def create_file(file: FileCreate):
    db = SessionLocal()
    try:
        db_file = FileMetadata(
            name=file.name,
            path=file.path,
            size=file.size,
            meta=json.dumps(file.meta)
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        logger.info(f"File created: {file.name}")
        return {"id": db_file.id, "name": db_file.name}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.get("/files")
async def list_files():
    db = SessionLocal()
    try:
        files = db.query(FileMetadata).all()
        return [
            {
                "id": f.id,
                "name": f.name,
                "path": f.path,
                "size": f.size,
                "created_at": f.created_at.isoformat(),
                "meta": json.loads(f.meta) if f.meta else {}
            }
            for f in files
        ]
    finally:
        db.close()

@app.get("/files/{file_id}")
async def get_file(file_id: int):
    db = SessionLocal()
    try:
        file = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return {
            "id": file.id,
            "name": file.name,
            "path": file.path,
            "size": file.size,
            "created_at": file.created_at.isoformat(),
            "meta": json.loads(file.meta) if file.meta else {}
        }
    finally:
        db.close()

@app.get("/files/{file_id}/download")
async def download_file(file_id: int):
    db = SessionLocal()
    try:
        file = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = Path(file.path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        return FileResponse(file_path, filename=file.name)
    finally:
        db.close()

# @app.get("/files/path/{file_path}")
# async def get_file_by_path(file_path: str):
#     path = Path(file_path)
#     if not path.exists():
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(path, filename=path.name)


# Datasets endpoints
@app.post("/datasets")
async def create_dataset(dataset: DatasetCreate):
    db = SessionLocal()
    try:
        db_dataset = Dataset(
            name=dataset.name,
            file_path=dataset.file_path,
            rows=dataset.rows,
            columns=dataset.columns,
            meta=json.dumps(dataset.meta)
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        logger.info(f"Dataset created: {dataset.name}")
        return {"id": db_dataset.id, "name": db_dataset.name}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()
    

def dataset_dict(dataset):
    return {
        "id": dataset.id,
        "name": dataset.name,
        "file_path": dataset.file_path,
        "rows": dataset.rows,
        "columns": dataset.columns,
        "created_at": dataset.created_at.isoformat(),
        "meta": json.loads(dataset.meta) if dataset.meta else {}
    }

@app.get("/datasets")
async def list_datasets():
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).all()
        return [
            dataset_dict(d)
            for d in datasets
        ]
    finally:
        db.close()

@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset_dict(dataset)
    finally:
        db.close()

@app.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: int):
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset_path = Path(dataset.file_path)
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found on disk")
        
        return FileResponse(dataset_path, filename=dataset.name + ".csv")
    finally:
        db.close()

# Models endpoints
@app.post("/models")
async def create_model(model: ModelCreate):
    db = SessionLocal()
    try:
        db_model = Model(
            name=model.name,
            model_type=model.model_type,
            version=model.version,
            file_path=model.file_path,
            dataset_id=model.dataset_id,
            accuracy=model.accuracy,
            meta=json.dumps(model.meta)
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        logger.info(f"Model created: {model.name}")
        return {"id": db_model.id, "name": db_model.name}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating model: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.get("/models")
async def list_models():
    db = SessionLocal()
    try:
        models = db.query(Model).all()
        return [
            {
                "id": m.id,
                "name": m.name,
                "model_type": m.model_type,
                "version": m.version,
                "file_path": m.file_path,
                "dataset_id": m.dataset_id,
                "accuracy": m.accuracy,
                "created_at": m.created_at.isoformat(),
                "meta": json.loads(m.meta) if m.meta else {}
            }
            for m in models
        ]
    finally:
        db.close()

@app.get("/models/{model_id}")
async def get_model(model_id: int):
    db = SessionLocal()
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return {
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "version": model.version,
            "file_path": model.file_path,
            "dataset_id": model.dataset_id,
            "accuracy": model.accuracy,
            "created_at": model.created_at.isoformat(),
            "meta": json.loads(model.meta) if model.meta else {}
        }
    finally:
        db.close()

@app.get("/models/{model_id}/download")
async def download_model(model_id: int):
    db = SessionLocal()
    try:
        model = db.query(Model).filter(Model.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        file_path = Path(model.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found on disk")
        
        return FileResponse(file_path, filename=f"{model.name}.pkl")
    finally:
        db.close()

# Runs endpoints
@app.post("/runs")
async def create_run(run: RunCreate):
    db = SessionLocal()
    try:
        db_run = Run(
            run_type=run.run_type,
            model_id=run.model_id,
            dataset_id=run.dataset_id,
            status=run.status,
            meta=json.dumps(run.meta)
        )
        db.add(db_run)
        db.commit()
        db.refresh(db_run)
        logger.info(f"Run created: {db_run.id} ({run.run_type})")
        return {"id": db_run.id}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating run: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

@app.get("/runs")
async def list_runs():
    db = SessionLocal()
    try:
        runs = db.query(Run).all()
        return [
            {
                "id": r.id,
                "run_type": r.run_type,
                "model_id": r.model_id,
                "dataset_id": r.dataset_id,
                "status": r.status,
                "result": json.loads(r.result) if r.result else None,
                "created_at": r.created_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "meta": json.loads(r.meta) if r.meta else {}
            }
            for r in runs
        ]
    finally:
        db.close()

@app.get("/runs/{run_id}")
async def get_run(run_id: int):
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "id": run.id,
            "run_type": run.run_type,
            "model_id": run.model_id,
            "dataset_id": run.dataset_id,
            "status": run.status,
            "result": json.loads(run.result) if run.result else None,
            "created_at": run.created_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "meta": json.loads(run.meta) if run.meta else {}
        }
    finally:
        db.close()

@app.patch("/runs/{run_id}")
async def update_run(run_id: int, run_update: RunUpdate):
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        if run_update.status is not None:
            run.status = run_update.status
        if run_update.result is not None:
            run.result = json.dumps(run_update.result)
        if run_update.completed_at is not None:
            run.completed_at = run_update.completed_at
        elif run_update.status == "completed":
            run.completed_at = datetime.now()
        
        db.commit()
        logger.info(f"Run updated: {run_id}")
        return {"id": run.id, "status": run.status}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating run: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()

# File upload endpoints
@app.post("/upload/dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        file_path = datasets_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Dataset uploaded: {file.filename}")
        return {
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["service"]["port"])
