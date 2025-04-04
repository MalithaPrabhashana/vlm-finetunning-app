import os
import traceback

from huggingface_hub import HfApi

from services.training import trained_models

# def save_model(model, app_name, hf_username, hf_token):
#     model_path = f"models/{app_name}_finetuned"
#     os.makedirs(model_path, exist_ok=True)
#     model.save_pretrained(model_path)
#     HfApi().upload_folder(
#         folder_path=model_path,
#         repo_id=f"{hf_username}/{app_name}_finetuned",
#         token=hf_token,
#     )
#     return {"message": "Model saved to Hugging Face", "model_path": model_path}


def save_model(task_id, app_name, hf_username, hf_token):
    model, tokenizer = trained_models.get(task_id)
    if model is None:
        raise ValueError(f"No model found for task ID {task_id}")

    model_path = f"models/{app_name}_finetuned"
    os.makedirs(model_path, exist_ok=True)
    print(model_path)

    # model.save_pretrained(model_path)
    model.save_pretrained_merged(
        "model_path",
        tokenizer,
        save_method="q4_k_m",
    )

    print("Model saved locally")

    HfApi().upload_folder(
        folder_path=model_path,
        repo_id=f"{hf_username}/{app_name}_finetuned",
        token=hf_token,
    )
    return {"message": "Model saved to Hugging Face", "model_path": model_path}


def save_gguf_model(
    task_id: str, output_dir: str = "gguf_models", quant_method: str = "q4_k_m"
):
    if task_id not in trained_models:
        raise ValueError(f"No model/tokenizer found for task ID {task_id}")

    model, tokenizer = trained_models[task_id]
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, task_id.replace("-", "_"))
    os.makedirs(save_path, exist_ok=True)  # Ensure the save_path directory exists

    try:
        model.save_pretrained_merged(save_path)
        model.save_pretrained_gguf(save_path)
        return {
            "message": "Model saved locally in GGUF format",
            "output_path": save_path,
            "quantization": quant_method,
        }
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        traceback.print_exc()
        return {"message": "Model not saved locally in GGUF format", "error": str(e)}
