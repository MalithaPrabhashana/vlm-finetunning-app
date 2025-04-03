import os
from huggingface_hub import HfApi

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


from services.training import trained_models

def save_model(task_id, app_name, hf_username, hf_token):
    model = trained_models.get(task_id)
    if model is None:
        raise ValueError(f"No model found for task ID {task_id}")

    model_path = f"models/{app_name}_finetuned"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)

    HfApi().upload_folder(
        folder_path=model_path,
        repo_id=f"{hf_username}/{app_name}_finetuned",
        token=hf_token,
    )
    return {"message": "Model saved to Hugging Face", "model_path": model_path}
