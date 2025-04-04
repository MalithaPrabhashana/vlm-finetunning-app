import os
import shutil
import uuid
from typing import Dict

import torch
from datasets import load_dataset
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from huggingface_hub import HfApi
from PIL import Image
from pydantic import BaseModel
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

app = FastAPI()
AVAILABLE_MODELS = [
    "unsloth/Llama-3.2-Vision",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    "unsloth/Pixtral",
]
task_status: Dict[str, dict] = {}


# Input format defined
class FineTuningRequest(BaseModel):
    model_name: str
    dataset_path: str
    app_name: str


class InferenceRequest(BaseModel):
    app_name: str


class SaveModelRequest(BaseModel):
    app_name: str
    hf_username: str
    hf_token: str


# Dataset upload path
def load_dataset_with_size(dataset_name: str, split):
    dataset = load_dataset(dataset_name, split=f"train[:{split}]")
    return dataset


instruction = (
    "You are an expert radiographer. Describe accurately what you see in this image."
)


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["caption"]}]},
    ]
    return {"messages": conversation}


class ProgressCallback(TrainerCallback):
    def __init__(self, task_id: str, total_steps: int):
        self.task_id = task_id
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        progress = int((state.global_step / self.total_steps) * 100)
        task_status[self.task_id] = {
            "status": "RUNNING",
            "progress": progress,
            "error": None,
        }

    def on_train_end(self, args, state, control, **kwargs):
        task_status[self.task_id] = {
            "status": "COMPLETED",
            "progress": 100,
            "error": None,
        }

    def on_train_begin(self, args, state, control, **kwargs):
        task_status[self.task_id] = {"status": "RUNNING", "progress": 0, "error": None}


def train_model(model_name: str, task_id: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    task_status[task_id] = {"status": "RUNNING", "progress": 0, "error": None}

    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

        dataset = load_dataset_with_size("unsloth/Radiology_mini", 10)
        converted_dataset = [convert_to_conversation(sample) for sample in dataset]

        FastVisionModel.for_training(model)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=30,
                # num_train_epochs = 1, # Set this instead of max_steps for full training runs
                learning_rate=2e-4,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=2048,
            ),
        )
        total_steps = trainer.args.max_steps
        trainer.add_callback(ProgressCallback(task_id, total_steps))
        trainer.train()

        # Ensure final status is set (though callback should handle this)
        task_status[task_id] = {
            "status": "COMPLETED",
            "progress": 100,
            "error": None,
            "model": model,
        }
    except Exception as e:
        task_status[task_id] = {
            "status": "FAILED",
            "progress": task_status[task_id]["progress"],
            "error": str(e),
        }


def save_model(model, app_name, hf_username, hf_token):
    model_path = f"models/{app_name}_finetuned"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    hf_api = HfApi()
    hf_api.upload_folder(
        folder_path=model_path,
        repo_id=f"{hf_username}/{app_name}_finetuned",
        token=hf_token,
    )
    return {"message": "Model saved to Hugging Face", "model_path": model_path}


# ----------------------API DEFINED---------------------- #
# Endpoint to get the list of available models
@app.get("/models")
def get_models():
    return {"models": AVAILABLE_MODELS}


# Endpoint to get the list of available datasets
@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), user_id: str = Form(...)):
    user_folder = f"datasets/{user_id}"
    os.makedirs(user_folder, exist_ok=True)
    upload_path = f"{user_folder}/{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "path": upload_path}


# Endpoint to start fine-tuning
@app.post("/start-finetuning")
async def start_finetuning(
    request: FineTuningRequest, background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        train_model, "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit", task_id
    )
    return {"task_id": task_id, "status": "STARTED"}


@app.get("/status/{task_id}")
def check_status(task_id: str):
    status = task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "status": status["status"],
        "progress": status["progress"],
        "error": status["error"] if status["error"] else None,
    }


# Endpoint to save the fine-tuned model to Hugging Face
@app.post("/save-model")
def save_model_req(request: SaveModelRequest):
    status = task_status.get(
        request.app_name
    )  # Note: This assumes app_name is task_id; adjust if needed
    if not status or "model" not in status:
        raise HTTPException(
            status_code=404, detail="Model not found or training not completed"
        )
    return save_model(
        status["model"], request.app_name, request.hf_username, request.hf_token
    )


# Endpoint for inference
@app.post("/inference")
async def inference(request: InferenceRequest):
    model_path = f"models/finetuned"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Fine-tuned model not found")

    model, tokenizer = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)

    image = "workspace/test.jpg"
    # image = Image.open(file.file).convert("RGB")
    instruction = "Is there something interesting about this image?"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    output = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=64,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}
