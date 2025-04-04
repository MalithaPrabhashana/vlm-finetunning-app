import time
import pynvml
from datasets import load_dataset
from fastapi import HTTPException
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from utils.dataset_utils import convert_to_conversation

AVAILABLE_MODELS = [
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    "unsloth/Pixtral-12B-2409",
]
task_status = {}
trained_models = {}

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

class ProgressCallback(TrainerCallback):
    def __init__(self, task_id: str, total_steps: int):
        self.task_id = task_id
        self.total_steps = total_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        progress = int((state.global_step / self.total_steps) * 100)
        task_status[self.task_id] = {
            "status": "RUNNING",
            "progress": progress,
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
            "learning_rate": (
                state.log_history[-1].get("learning_rate")
                if state.log_history
                else None
            ),
            "epoch": state.epoch,
            "error": None,
        }

    def on_train_begin(self, args, state, control, **kwargs):
        task_status[self.task_id] = {
            "status": "RUNNING",
            "progress": 0,
            "loss": None,
            "learning_rate": None,
            "epoch": 0,
            "error": None,
        }

    def on_train_end(self, args, state, control, **kwargs):
        task_status[self.task_id].update({"status": "COMPLETED", "progress": 100})


def train_model(model_name: str, task_id: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model name")

    task_status[task_id] = {"status": "RUNNING", "progress": 0, "error": None}
    results: dict = {}

    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name, load_in_4bit=True, use_gradient_checkpointing="unsloth"
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
        )

        dataset = load_dataset("unsloth/Radiology_mini", split="train[:10]")
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
                max_steps=20,
                learning_rate=2e-4,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                output_dir="outputs",
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=2048,
                report_to="none",
            ),
        )

        # Extract table parameters
        results["Learning Rate"] = trainer.args.learning_rate
        results["Batch Size"] = trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps
        results["Training Steps"] = trainer.args.max_steps
        results["Mixed Precision"] = "BF16" if trainer.args.bf16 else "FP16"
        results["Sequence Length"] = trainer.args.max_seq_length

        # Measure training time
        start_time = time.time()
        trainer.add_callback(ProgressCallback(task_id, trainer.args.max_steps))
        trainer.train()
        training_time = (time.time() - start_time) / 3600
        results["Training Time (hrs)"] = training_time

        # Measure GPU usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_usage = (mem_info.used / mem_info.total) * 100
        results["GPU Usage (%)"] = gpu_usage

        task_status[task_id] = {"status": "COMPLETED", "progress": 100, "error": None}
        trained_models[task_id] = (model, tokenizer)

    except Exception as e:
        task_status[task_id] = {"status": "FAILED", "progress": 0, "error": str(e)}
