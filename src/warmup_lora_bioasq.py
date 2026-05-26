import os
import gc
import json
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

BIOASQ_TYPES = ["factoid", "yesno", "summary", "list"]


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, origin_dataset, tokenizer, args):
        max_length = args.block_size
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for data in origin_dataset:
            prompt_ids = prompt_template.get_prompt(
                tokenizer=tokenizer,
                question=data["question"],
                passages=None,
                answer=None,
                with_cot=False,
            )

            answer = data["answer"]
            if not answer.endswith("."):
                answer += "."
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_ids.append(tokenizer.eos_token_id)

            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            pad_length = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            labels = input_ids + [self.ignored_id] * pad_length
            input_ids += [pad_token_id] * pad_length

            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }


def train_for_type(qtype, model, tokenizer, args):
    """Train warmup LoRA weights for a specific BioASQ question type."""
    data_path = os.path.join(ROOT_DIR, "warmup", "data", "direct", f"bioasq_{qtype}.json")
    if not os.path.exists(data_path):
        print(f"WARNING: No warmup data found for {qtype} at {data_path}, skipping.")
        return

    with open(data_path, "r") as fin:
        dataset = json.load(fin)
    random.shuffle(dataset)
    print(f"\n{'='*60}")
    print(f"Training warmup for BioASQ type: {qtype} ({len(dataset)} samples)")
    print(f"{'='*60}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True

    train_data = TrainingData(dataset, tokenizer, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, peft_model.device),
        shuffle=False,
    )

    model_parameters = filter(lambda p: p.requires_grad, peft_model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    logging_step = 10
    losses = []
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs} [{qtype}]")):
            optimizer.zero_grad()
            outputs = peft_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if step % logging_step == 0:
                print(f"[{qtype}] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                losses.append(loss.item())

    save_path = os.path.join(
        ROOT_DIR,
        "warmup",
        "lora_base_weight",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        f"bioasq_{qtype}",
    )
    os.makedirs(save_path, exist_ok=True)
    peft_model.save_pretrained(save_path)
    with open(os.path.join(save_path, "training_config.json"), "w") as fout:
        json.dump({**vars(args), "question_type": qtype, "num_samples": len(dataset)}, fout, indent=4)
    print(f"Saved warmup weights for {qtype} to {save_path}")

    # Unload LoRA so we can reuse the base model for the next type
    peft_model = peft_model.unload()
    torch.cuda.empty_cache()
    gc.collect()


def main(args):
    model, tokenizer, _generation_config = get_model(args.model_name)

    if args.question_type:
        types_to_train = [args.question_type]
    else:
        types_to_train = BIOASQ_TYPES

    for qtype in types_to_train:
        train_for_type(qtype, model, tokenizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--question_type", type=str, default=None,
                        choices=BIOASQ_TYPES,
                        help="Train warmup for a specific question type. If not set, trains all types.")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=3000)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)

    args = parser.parse_args()
    main(args)
