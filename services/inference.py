import os

import torch
from fastapi import HTTPException
from PIL import Image
from transformers import TextStreamer
from unsloth import FastVisionModel


def run_inference(request):
    model_path = f"models/finetuned"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Fine-tuned model not found")

    model, tokenizer = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
    image = "workspace/test.jpg"
    instruction = "Is there something interesting about this image?"

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    inputs = tokenizer(image, input_text, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output = model.generate(
        **inputs, streamer=TextStreamer(tokenizer), max_new_tokens=64
    )

    return {"response": tokenizer.decode(output[0], skip_special_tokens=True)}
