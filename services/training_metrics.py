import torch

def print_training_summary(trainer):
    # Get initial GPU memory usage
    start_gpu_memory = round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3)
    max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    runtime_seconds = trainer.state.log_history[-1]['train_runtime']
    runtime_minutes = round(runtime_seconds / 60, 2)

    print(f"TRAINING SUMMARY")
    print(f"Runtime: {runtime_seconds} sec ({runtime_minutes} min)")
    print(f"Peak reserved memory: {used_memory} GB")
    print(f"Memory used for LoRA: {used_memory_for_lora} GB")
    print(f"Memory usage %: {used_percentage}%")
    print(f"LoRA memory %: {lora_percentage}%")

    return {
        "train_runtime_seconds": runtime_seconds,
        "train_runtime_minutes": runtime_minutes,
        "peak_memory_gb": used_memory,
        "lora_memory_gb": used_memory_for_lora,
        "memory_usage_percent": used_percentage,
        "lora_memory_percent": lora_percentage,
    }