from transformers import TrainerCallback
from services.training import task_status

class ProgressCallback(TrainerCallback):
    def __init__(self, task_id: str, total_steps: int):
        self.task_id = task_id
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        progress = int((state.global_step / self.total_steps) * 100)
        task_status[self.task_id] = {"status": "RUNNING", "progress": progress, "error": None}

    def on_train_end(self, args, state, control, **kwargs):
        task_status[self.task_id] = {"status": "COMPLETED", "progress": 100, "error": None}

    def on_train_begin(self, args, state, control, **kwargs):
        task_status[self.task_id] = {"status": "RUNNING", "progress": 0, "error": None}
