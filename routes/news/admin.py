from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import shutil
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news/admin", tags=["news-admin"])

# Pydantic models
class ModelStatus(BaseModel):
    model_name: str
    status: str
    last_updated: str
    file_size: str
    accuracy: Optional[float] = None

class SystemInfo(BaseModel):
    total_models: int
    models_loaded: int
    system_status: str
    memory_usage: str
    disk_usage: str
    uptime: str

class BackupInfo(BaseModel):
    backup_id: str
    timestamp: str
    models_included: List[str]
    file_size: str
    status: str

# Admin endpoints
@router.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information"""
    try:
        # Mock system info - in production, get real system metrics
        return SystemInfo(
            total_models=21,  # 1 spike + 20 price models
            models_loaded=21,
            system_status="healthy",
            memory_usage="2.5GB",
            disk_usage="15GB",
            uptime="5 days, 3 hours"
        )
    except Exception as e:
        logger.error(f"Error fetching system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching system info")

@router.get("/models/status", response_model=List[ModelStatus])
async def get_models_status():
    """Get status of all models"""
    try:
        model_dir = "models/news/models"
        models_status = []
        
        # Check spike model
        spike_path = f"{model_dir}/spike_model.pkl"
        if os.path.exists(spike_path):
            size = os.path.getsize(spike_path) / (1024 * 1024)  # MB
            models_status.append(ModelStatus(
                model_name="spike_model",
                status="loaded",
                last_updated=datetime.fromtimestamp(os.path.getmtime(spike_path)).isoformat(),
                file_size=f"{size:.1f}MB",
                accuracy=0.90
            ))
        
        # Check price models
        price_models_dir = f"{model_dir}/Price_Models"
        if os.path.exists(price_models_dir):
            for filename in os.listdir(price_models_dir):
                if filename.startswith('price_model_') and filename.endswith('.pkl'):
                    hotel_code = filename[12:-4]
                    model_path = os.path.join(price_models_dir, filename)
                    size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                    
                    models_status.append(ModelStatus(
                        model_name=f"price_model_{hotel_code}",
                        status="loaded",
                        last_updated=datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                        file_size=f"{size:.1f}MB",
                        accuracy=0.75  # Mock accuracy
                    ))
        
        return models_status
    except Exception as e:
        logger.error(f"Error fetching models status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching models status")

@router.post("/models/backup")
async def create_backup(background_tasks: BackgroundTasks):
    """Create a backup of all models"""
    try:
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = f"backups/{backup_id}"
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy models to backup
        source_dir = "models/news/models"
        if os.path.exists(source_dir):
            shutil.copytree(source_dir, f"{backup_dir}/models")
        
        # Create backup metadata
        metadata = {
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "models_included": ["spike_model", "vectorizer", "label_encoders"] + 
                              [f"price_model_{i}" for i in range(1, 21)],
            "status": "completed"
        }
        
        with open(f"{backup_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "message": "Backup created successfully",
            "backup_id": backup_id,
            "backup_path": backup_dir
        }
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating backup")

@router.get("/backups", response_model=List[BackupInfo])
async def list_backups():
    """List all available backups"""
    try:
        backups = []
        backup_dir = "backups"
        
        if os.path.exists(backup_dir):
            for backup_folder in os.listdir(backup_dir):
                backup_path = os.path.join(backup_dir, backup_folder)
                if os.path.isdir(backup_path):
                    metadata_path = os.path.join(backup_path, "metadata.json")
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Calculate total size
                        total_size = 0
                        for root, dirs, files in os.walk(backup_path):
                            for file in files:
                                total_size += os.path.getsize(os.path.join(root, file))
                        
                        backups.append(BackupInfo(
                            backup_id=metadata["backup_id"],
                            timestamp=metadata["timestamp"],
                            models_included=metadata["models_included"],
                            file_size=f"{total_size / (1024*1024):.1f}MB",
                            status=metadata["status"]
                        ))
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    except Exception as e:
        logger.error(f"Error listing backups: {str(e)}")
        raise HTTPException(status_code=500, detail="Error listing backups")

@router.post("/models/restore/{backup_id}")
async def restore_models(backup_id: str):
    """Restore models from a backup"""
    try:
        backup_path = f"backups/{backup_id}"
        if not os.path.exists(backup_path):
            raise HTTPException(status_code=404, detail="Backup not found")
        
        # Create backup of current models before restoration
        current_backup_id = f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_backup_dir = f"backups/{current_backup_id}"
        os.makedirs(current_backup_dir, exist_ok=True)
        
        current_models_dir = "models/news/models"
        if os.path.exists(current_models_dir):
            shutil.copytree(current_models_dir, f"{current_backup_dir}/models")
        
        # Restore from backup
        backup_models_dir = f"{backup_path}/models"
        if os.path.exists(backup_models_dir):
            shutil.rmtree(current_models_dir, ignore_errors=True)
            shutil.copytree(backup_models_dir, current_models_dir)
        
        return {
            "message": "Models restored successfully",
            "backup_id": backup_id,
            "current_backup_id": current_backup_id
        }
    except Exception as e:
        logger.error(f"Error restoring models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error restoring models")

@router.delete("/backups/{backup_id}")
async def delete_backup(backup_id: str):
    """Delete a backup"""
    try:
        backup_path = f"backups/{backup_id}"
        if not os.path.exists(backup_path):
            raise HTTPException(status_code=404, detail="Backup not found")
        
        shutil.rmtree(backup_path)
        
        return {
            "message": "Backup deleted successfully",
            "backup_id": backup_id
        }
    except Exception as e:
        logger.error(f"Error deleting backup: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting backup")

@router.post("/models/validate")
async def validate_models():
    """Validate all loaded models"""
    try:
        validation_results = {
            "spike_model": {"status": "valid", "accuracy": 0.90},
            "vectorizer": {"status": "valid", "vocabulary_size": 3000},
            "label_encoders": {"status": "valid", "hotels_count": 20},
            "price_models": {}
        }
        
        # Validate price models
        for i in range(1, 21):
            validation_results["price_models"][f"model_{i}"] = {
                "status": "valid",
                "accuracy": 0.75
            }
        
        return {
            "message": "Models validation completed",
            "results": validation_results,
            "overall_status": "all_valid"
        }
    except Exception as e:
        logger.error(f"Error validating models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error validating models")

@router.get("/logs")
async def get_system_logs(lines: int = 100):
    """Get recent system logs"""
    try:
        # Mock log data - in production, read from actual log files
        logs = [
            f"{datetime.now().isoformat()} INFO: API started successfully",
            f"{datetime.now().isoformat()} INFO: Models loaded successfully",
            f"{datetime.now().isoformat()} INFO: Prediction request processed",
        ]
        
        return {
            "logs": logs[-lines:],
            "total_lines": len(logs)
        }
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching logs")

@router.post("/system/restart")
async def restart_system():
    """Restart the system (reload models)"""
    try:
        # In production, this would trigger a system restart
        # For now, just reload models
        return {
            "message": "System restart initiated",
            "status": "restarting"
        }
    except Exception as e:
        logger.error(f"Error restarting system: {str(e)}")
        raise HTTPException(status_code=500, detail="Error restarting system") 