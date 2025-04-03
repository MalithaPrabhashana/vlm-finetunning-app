from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from api.models import FineTuningRequest, InferenceRequest, SaveModelRequest
from services.training import train_model, task_status, AVAILABLE_MODELS
from services.save import save_model
from services.inference import run_inference
import uuid
import os
import shutil

router = APIRouter()

@router.get("/models")
def get_models():
    return {"models": AVAILABLE_MODELS}

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), user_id: str = Form(...)):
    user_folder = f"datasets/{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    upload_path = f"{user_folder}/{file.filename}"
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": upload_path}

@router.post("/start-finetuning")
async def start_finetuning(request: FineTuningRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(train_model, request.model_name, task_id)
    return {"task_id": task_id, "status": "STARTED"}

@router.get("/status/{task_id}")
def check_status(task_id: str):
    status = task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": status["status"],
        "progress": status["progress"],
        "error": status["error"] if status["error"] else None
    }

@router.post("/save-model")
def save_model_req(request: SaveModelRequest):
    status = task_status.get(request.app_name)
    if not status or "model" not in status:
        raise HTTPException(status_code=404, detail="Model not found or training not completed")
    return save_model(status["model"], request.app_name, request.hf_username, request.hf_token)

@router.post("/inference")
async def inference(request: InferenceRequest):
    return run_inference(request)
