import argparse
import gc
import json
import os

import prompt_template
import torch
from peft import ArrowConfig, PeftModel, create_arrow_model
from prompt_template import get_task_instruction
from root_dir_path import ROOT_DIR
from tqdm import tqdm
from utils import BaseDataset, evaluate, get_model, load_data, predict, read_complete


def has_non_empty_answer(answer):
    """Return True when at least one non-empty gold answer string exists."""
    flat = BaseDataset.flatten_ground_truth(answer)
    return any(v is not None and str(v).strip() for v in flat)


def validate_labels_or_raise(dataset, filename):
    """Fail fast when strict evaluation is requested but labels are missing."""
    missing = []
    for i, sample in enumerate(dataset):
        if not has_non_empty_answer(sample.get("answer", None)):
            missing.append(i)

    if not missing:
        return

    preview = ", ".join(str(i) for i in missing[:10])
    raise ValueError(
        "--require_labels is enabled, but missing/empty gold answers were found "
        f"in split '{filename}': {len(missing)}/{len(dataset)} samples. "
        f"First missing indices: [{preview}]. "
        "This commonly indicates an unlabeled split (e.g., BioASQ phase-A). "
        "Use a labeled split or rerun without --require_labels."
    )


def main(args):
    data_list = load_data(args.dataset, args.data_type, args.augment_model)
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    if args.with_cot:
        prompt_template.get_fewshot(args.dataset)

    cot_name = "cot" if args.with_cot else "direct"
    load_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
    )
    output_root_dir = os.path.join(
        ROOT_DIR,
        "output",
        args.model_name,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}",
        args.dataset,
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}_{cot_name}",
        f"aug_model={args.augment_model}",
        args.inference_method,
    )
    for filename, fulldata in data_list:
        filename = filename.split(".")[0]
        # Simpler usage for boolean flags (action="store_true"):
        # They are guaranteed to exist on args (defaulting to False).
        if args.require_labels:
            validate_labels_or_raise(fulldata, filename)

        print(f"### Solving {filename} ###")
        output_dir = os.path.join(output_root_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            json.dump(vars(args), fout, indent=4)

        predict_file = os.path.join(output_dir, "predict.json")
        ret, start_with = read_complete(predict_file)

        fulldata = (
            fulldata[start_with:]
            if args.sample == -1
            else fulldata[start_with : args.sample]
        )
        for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
            test_id = test_id + start_with
            assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

            question = data["question"]
            passages = data["passages"]
            answer = data["answer"]

            # Resolve task-specific instruction if templates are enabled
            task_instruction = None
            if args.use_task_templates:
                q_type = data.get("type", None)
                if q_type:
                    task_instruction = get_task_instruction(
                        q_type, args.inference_method
                    )

            def get_pred(model, psgs):
                text = predict(
                    model,
                    tokenizer,
                    generation_config,
                    question,
                    with_cot=args.with_cot,
                    passages=psgs,
                    task_instruction=task_instruction,
                )
                pred = {
                    "test_id": test_id,
                    "question": question,
                    "answer": answer,
                    "text": text,
                }
                pred.update(evaluate(text, answer, args.with_cot))
                return pred

            def print_lora_debug(m, step):
                print(f"=== {step} ===")
                if hasattr(m, "peft_config"):
                    print("Adapters in config:", list(m.peft_config.keys()))
                    for adp in ["0", "1", "2", "merge"]:
                        print(f"Adapter {adp} exists:", adp in m.peft_config)
                else:
                    print("PEFT config missing. Model is raw.")
                for adp in ["0", "1", "2", "merge"]:
                    s = 0.0
                    for n, p in m.named_parameters():
                        if (
                            f".lora_A.{adp}." in n
                            or f".lora_B.{adp}." in n
                            or f".lora_embedding_A.{adp}" in n
                            or f".lora_embedding_B.{adp}" in n
                        ):
                            s += p.sum().item()
                    print(f"Summed weight of {adp}: {s}")

            if args.inference_method == "icl":
                ret.append(get_pred(model, psgs=passages))
            else:
                if args.use_arrow:
                    task_specific_adapter_paths = []
                    for pid in range(len(passages)):
                        adapter_path = os.path.join(
                            load_adapter_path,
                            filename,
                            f"data_{test_id}",
                            f"passage_{pid}",
                        )
                        task_specific_adapter_paths.append(adapter_path)

                    arrow_config = ArrowConfig(
                        top_k=min(args.arrow_top_k, len(passages)),
                        router_temperature=1.0,
                        rng_seed=42,
                    )

                    model = create_arrow_model(
                        base_model=model,
                        task_specific_adapter_paths=task_specific_adapter_paths,
                        arrow_config=arrow_config,
                    )
                    print_lora_debug(model, "After create_arrow_model")

                    ret.append(
                        get_pred(
                            model,
                            psgs=None if args.inference_method == "prag" else passages,
                        )
                    )

                    model = model.unload()
                    # We can unconditionally delete peft_config here just like in the ties method.
                    # The previous hasattr was just to prevent an AttributeError if the model was already raw,
                    # but since create_arrow_model always adds it, it's perfectly safe to delete directly!
                    del model.peft_config
                    print_lora_debug(model, "After model.unload")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    for pid in range(len(passages)):
                        adapter_path = os.path.join(
                            load_adapter_path,
                            filename,
                            f"data_{test_id}",
                            f"passage_{pid}",
                        )
                        if pid == 0:
                            model = PeftModel.from_pretrained(
                                model,
                                adapter_path,
                                adapter_name="0",
                                is_trainable=False,
                            )
                        else:
                            model.load_adapter(adapter_path, adapter_name=str(pid))

                    print_lora_debug(model, "After model.load")
                    # merge
                    model.add_weighted_adapter(
                        adapters=[str(i) for i in range(len(passages))],
                        weights=[1.0 / len(passages)] * len(passages),
                        adapter_name="merge",
                        combination_type="cat",
                    )
                    print_lora_debug(model, "After model.add_weighted_adapter")
                    model.set_adapter("merge")
                    print_lora_debug(model, "After model.set_adapter")

                    ret.append(
                        get_pred(
                            model,
                            psgs=None if args.inference_method == "prag" else passages,
                        )
                    )

                    model.delete_adapter("merge")
                    print_lora_debug(model, "After model.delete_adapter")

                    model = model.unload()
                    del model.peft_config
                    print_lora_debug(model, "After model.unload")

                    torch.cuda.empty_cache()
                    gc.collect()

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

        ##### Evaluating #####
        metrics = ["em", "f1", "prec", "recall"]
        evaluable = [d for d in ret if d.get("is_evaluable", "1") == "1"]
        ret_str = f"evaluated_samples\t{len(evaluable)}/{len(ret)}\n"
        for met in metrics:
            if not evaluable:
                ret_str += f"{met}\tNA\n"
                continue

            vals = [
                float(d[met])
                for d in evaluable
                if d.get(met) not in (None, "", "nan", "NaN")
            ]
            if not vals:
                ret_str += f"{met}\tNA\n"
                continue

            acc = round(sum(vals) / len(vals), 4)
            ret_str += f"{met}\t{acc}\n"
        ret_str += "\n" + json.dumps(vars(args), indent=4)
        with open(os.path.join(output_dir, "result.txt"), "w") as fout:
            fout.write(ret_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--with_cot", action="store_true")
    parser.add_argument("--sample", type=int, default=-1)  # -1 means all
    parser.add_argument("--augment_model", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument(
        "--inference_method",
        type=str,
        required=True,
        choices=["icl", "prag", "combine"],
    )
    parser.add_argument(
        "--require_labels",
        action="store_true",
        help="Fail fast if any sample has missing/empty gold answer labels",
    )
    parser.add_argument(
        "--use_task_templates",
        action="store_true",
        help="Prepend task-specific instruction (factoid/yesno/list/summary) to the prompt",
    )
    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)

    # Arrow
    parser.add_argument("--use_arrow", action="store_true")
    parser.add_argument("--arrow_top_k", type=int, default=3)

    args = parser.parse_args()
    assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
    if args.augment_model is None:
        args.augment_model = args.model_name
    print(args)
    main(args)
