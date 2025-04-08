import os
import torch
import gc
import unsloth
import pandas as pd
from datasets import load_dataset
from transformers import TrainerCallback
from unsloth import FastVisionModel, is_bf16_supported
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TextStreamer
import numpy as np

# ----------- Settings ----------- #
EXCEL_PATH = "hyperparameters.xlsx"
DATASET_NAME = "unsloth/Radiology_mini"
DATASET_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

instruction = (
    "You are an expert radiographer. Describe accurately what you see in this image."
)

# ----------- Functions ----------- #


def convert_to_conversation(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["caption"]}],
            },
        ]
    }


def evaluate_model(sentence_model, model, tokenizer, dataset):
    generated_texts = []
    ground_truths = []

    for sample in dataset:
        image = sample["messages"][0]["content"][1]["image"]
        gt_caption = sample["messages"][1]["content"][0]["text"]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": gt_caption}],
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image, 
            input_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(DEVICE)

        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        output = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.5, min_p = 0.1)

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(generated)
        ground_truths.append(gt_caption)

    # Embedding and cosine similarity
    gen_embeddings = sentence_model.encode(generated_texts, convert_to_tensor=True)
    gt_embeddings = sentence_model.encode(ground_truths, convert_to_tensor=True)
    similarities = util.cos_sim(gen_embeddings, gt_embeddings)
    diagonal_scores = similarities.diag()
    mean_score = diagonal_scores.mean().item()
    return round(mean_score, 4)


def fine_tune_model(params: dict):
    print(
        f"Training model: {params['model_name']} with LR={params['learning_rate']}, Steps={params['max_steps']}"
    )

    model, tokenizer = FastVisionModel.from_pretrained(
        params["model_name"],
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
    )

    dataset = load_dataset(DATASET_NAME, split=f"train[:{DATASET_SIZE}]")
    dataset = [convert_to_conversation(sample) for sample in dataset]
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=int(params["batch_size"]),
            gradient_accumulation_steps=4,
            max_steps=int(params["max_steps"]),
            learning_rate=float(params["learning_rate"]),
            warmup_steps=5,
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

    trainer.train()

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    accuracy = evaluate_model(sentence_model, model, tokenizer, dataset)

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return accuracy


# ----------- Main Execution ----------- #
def main():
    df = pd.read_excel(EXCEL_PATH)

    for idx, row in df.iterrows():
        if pd.notna(row.get("accuracy")):
            print(f"Skipping row {idx}: already trained.")
            continue

        try:
            hyperparams = {
                "model_name": row["model_name"],
                "learning_rate": row["learning_rate"],
                "batch_size": row["batch_size"],
                "max_steps": row["max_steps"],
            }

            acc = fine_tune_model(hyperparams)
            df.at[idx, "accuracy"] = acc
            print(f"Row {idx} trained. Accuracy: {acc}")

        except Exception as e:
            print(f"Error in row {idx}: {e}")
            df.at[idx, "accuracy"] = f"Error: {e}"

        df.to_excel(EXCEL_PATH, index=False)

    print("All fine-tuning completed!")


if __name__ == "__main__":
    main()
