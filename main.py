from fastapi import FastAPI, UploadFile, File, HTTPException
from unsloth import FastVisionModel
from trl import SFTTrainer
from transformers import TrainingArguments
import shutil
import os

app = FastAPI()

# List of Unsloth-supported VLMs
AVAILABLE_MODELS = ["unsloth/Llama-3.2-Vision", "unsloth/Qwen2-VL", "unsloth/Pixtral"]

@app.get("/models")
def get_models():
    return {"models": AVAILABLE_MODELS}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    upload_path = f"temp/{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": upload_path}

@app.post("/start-finetuning")
async def start_finetuning(model_name: str, dataset_path: str, app_name: str, hf_token: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")
    
    # Simulate async task (replace with Celery in production)
    model, tokenizer = FastVisionModel.from_pretrained(model_name, load_in_4bit=True)
    # Configure LoRA, load dataset, set up trainer (as above)
    # trainer.train()
    
    return {"task_id": "12345", "message": "Fine-tuning started"}

@app.get("/status/{task_id}")
def check_status(task_id: str):
    # Check task status (e.g., with Celery)
    return {"task_id": task_id, "status": "in_progress"}

@app.post("/save-model")
def save_model(app_name: str, hf_token: str):
    # Save and upload (as above)
    return {"message": f"Model {app_name}_finetuned saved to Hugging Face"}