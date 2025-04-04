from pydantic import BaseModel

class FineTuningRequest(BaseModel):
    model_name: str
    dataset_path: str
    app_name: str

class InferenceRequest(BaseModel):
    app_name: str

class SaveModelRequest(BaseModel):
    task_id: str
    app_name: str
    hf_username: str
    hf_token: str

class GGUFSaveRequest(BaseModel):
    task_id: str
    app_name: str
    quant_method: str = "q4_k_m"
    output_dir: str = "gguf_models"