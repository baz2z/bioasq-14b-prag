import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_model, model_generate
from root_dir_path import ROOT_DIR


random.seed(42)

BIOASQ_TYPES = ["yesno", "factoid", "list", "summary"]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def get_exact_answer(question):
    """Extract exact_answer for yesno/factoid/list; ideal_answer for summary.

    Returns
    -------
    answer : str | list[list[str]]
        - yesno   -> str  ("yes" / "no")
        - factoid -> list[list[str]]  (all synonyms kept)
        - list    -> list[list[str]]  (all synonyms kept)
        - summary -> str  (first ideal answer)
    """
    qtype = question.get("type")

    if qtype == "yesno":
        return question.get("exact_answer", "")

    elif qtype == "factoid":
        # exact_answer is list[list[str]] where inner lists are synonyms
        # Keep all synonyms
        exact = question.get("exact_answer", [])
        return [syns if isinstance(syns, list) else [syns] for syns in exact]

    elif qtype == "list":
        # Same structure as factoid – list of lists, keep all
        exact = question.get("exact_answer", [])
        return [items if isinstance(items, list) else [items] for items in exact]

    elif qtype == "summary":
        # No exact_answer for summary – use ideal_answer
        ia = question.get("ideal_answer", [])
        if isinstance(ia, list):
            return ia[0] if ia else ""
        return ia

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_bioasq_golden(data_path):
    """Load the BioASQ golden dataset and separate by question type."""
    with open(data_path, "r") as fin:
        dataset = json.load(fin)

    questions = dataset["questions"]

    type_to_dataset = {t: [] for t in BIOASQ_TYPES}

    for data in questions:
        qtype = data["type"]
        if qtype not in type_to_dataset:
            continue

        answer = get_exact_answer(data)
        if answer is None:
            continue

        transformed = {
            "test_id": len(type_to_dataset[qtype]),
            "question": data["body"],
            "answer": answer,
            "passages": [s.get("text", "") for s in data.get("snippets", [])],
            "original_id": data["id"],
            "documents": data.get("documents", []),
            "type": qtype,
        }

        # For yesno/factoid/list, also store the ideal_answer so that
        # inference can use it for the ideal-answer condition without
        # needing a separate dataset.  Summary already uses ideal_answer
        # as its primary answer, so no extra field needed there.
        if qtype != "summary":
            ia = data.get("ideal_answer", [])
            if isinstance(ia, list):
                ia = ia[0] if ia else ""
            transformed["ideal_answer"] = ia

        type_to_dataset[qtype].append(transformed)

    return type_to_dataset


# ---------------------------------------------------------------------------
# Augmentation helpers (rewrite + QA generation)
# ---------------------------------------------------------------------------

def get_rewrite(passage, model_name, model=None, tokenizer=None, generation_config=None):
    """Generate a rewrite for a passage."""
    rewrite_prompt = (
        "Rewrite the following passage. While keeping the entities, proper nouns, "
        "and key details such as names, locations, and terminology intact, create a "
        "new version of the text that expresses the same ideas in a different way. "
        "Make sure the revised passage is distinct from the original one, but "
        "preserves the core meaning and relevant information. Use different sentence "
        "structures and word choices while maintaining scientific accuracy.\n\n"
        "Original passage:\n{passage}\n\nRewritten passage:"
    )
    return model_generate(rewrite_prompt.format(passage=passage),
                          model, tokenizer, generation_config)


qa_prompt_template = (
    "I will provide a passage of text, and you need to generate three different "
    "questions based on the content of this passage. Each question should be "
    "answerable using the information provided in the passage. Additionally, "
    "please provide an appropriate answer for each question derived from the "
    "passage.\n"
    "You need to generate the question and answer in the following format:\n"
    "[\n"
    "    {{\n"
    '        "question": "What is the capital of France?",\n'
    '        "answer": "Paris",\n'
    '        "full_answer": "The capital of France is Paris."\n'
    "    }}\n"
    "]\n\n"
    "This list should have at least three elements. You only need to output "
    "this list in the above format.\n"
    "Passage:\n{passage}"
)


def fix_qa(qa):
    """Fix and validate QA pairs."""
    if isinstance(qa, list):
        if len(qa) >= 3:
            qa = qa[:3]
            for data in qa:
                if "question" not in data or "answer" not in data or "full_answer" not in data:
                    return False, qa
                if isinstance(data["answer"], list):
                    data["answer"] = ", ".join(data["answer"])
                if isinstance(data["answer"], int):
                    data["answer"] = str(data["answer"])
                if data["answer"] is None:
                    data["answer"] = "Unknown"
            return True, qa
    return False, qa


def get_qa(passage, model_name, model=None, tokenizer=None, generation_config=None):
    """Generate QA pairs for a passage."""

    def fix_json(output):
        # Generic JSON extraction that works across model families
        if "[" in output:
            output = output[output.find("["):]
        if "]" in output:
            output = output[:output.rfind("]") + 1]
        if output.endswith(","):
            output = output[:-1]
        if not output.endswith("]"):
            output += "]"
        return output

    try_times = 100
    prompt = qa_prompt_template.format(passage=passage)
    output = None

    while try_times:
        output = model_generate(prompt, model, tokenizer, generation_config)
        output = fix_json(output)
        try:
            qa = json.loads(output)
            ret, qa = fix_qa(qa)
            if ret:
                return qa
        except Exception:
            try_times -= 1
    return output


# ---------------------------------------------------------------------------
# Passage selection via MMR
# ---------------------------------------------------------------------------

def select_passages_mmr(question, passages, cross_encoder, sentence_model, topk=3):
    """Select diverse & relevant passages using Maximal Marginal Relevance."""
    if len(passages) <= topk:
        return passages

    scores = cross_encoder.predict(
        [(question, p) for p in passages]
    )
    embeddings = sentence_model.encode(passages)

    selected_indices = []
    remaining = list(range(len(passages)))

    # 1st passage: highest relevance
    best = max(remaining, key=lambda i: scores[i])
    selected_indices.append(best)
    remaining.remove(best)

    lam = 0.5
    for _ in range(min(topk - 1, len(remaining))):
        mmr = []
        for i in remaining:
            sims = [
                cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                for j in selected_indices
            ]
            mmr_score = lam * scores[i] - (1 - lam) * max(sims)
            mmr.append((i, mmr_score))
        best = max(mmr, key=lambda x: x[1])[0]
        selected_indices.append(best)
        remaining.remove(best)

    return [passages[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def _build_total(type_to_ret):
    """Merge per-type results into a single total list (sorted by type order)."""
    total = []
    for t in BIOASQ_TYPES:
        if t in type_to_ret:
            total.extend(type_to_ret[t])
    # Re-number test_id sequentially
    for i, item in enumerate(total):
        item["test_id"] = i
    return total


def save_datasets(type_to_ret, model_name):
    """Save augmented dataset to ``data_aug/bioasq/<model>/``."""
    out_dir = os.path.join(ROOT_DIR, "data_aug", "bioasq", model_name)
    os.makedirs(out_dir, exist_ok=True)

    for qtype, items in type_to_ret.items():
        out_path = os.path.join(out_dir, f"{qtype}.json")
        with open(out_path, "w") as fout:
            json.dump(items, fout, indent=4)
        print(f"  Saved {len(items)} {qtype} items -> {out_path}")

    total = _build_total(type_to_ret)
    with open(os.path.join(out_dir, "total.json"), "w") as fout:
        json.dump(total, fout, indent=4)
    print(f"  Saved {len(total)} total items -> {out_dir}/total.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("### Loading BioASQ golden dataset ###")
    type_to_dataset = load_bioasq_golden(args.data_path)

    for t in BIOASQ_TYPES:
        print(f"  {t}: {len(type_to_dataset.get(t, []))} questions")

    # Snippet ranking & diversity models
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L2-v2")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # LLM for rewrite + QA generation
    model, tokenizer, _ = get_model(args.model_name)
    generation_config = dict(
        max_new_tokens=512,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=0.7,
        top_k=50,
    )

    type_to_ret = {}  # collect augmented results per type

    for qtype in BIOASQ_TYPES:
        dataset = type_to_dataset.get(qtype, [])
        if not dataset:
            continue

        if args.sample > 0:
            dataset = dataset[: args.sample]

        print(f"\n### Processing {qtype} ({len(dataset)} questions) ###")

        total_passages = sum(min(args.top_k_snippets, len(d["passages"])) for d in dataset)
        pbar = tqdm(total=total_passages, desc=f"Augmenting {qtype}")
        ret = []

        for data in dataset:
            data["augment"] = []

            passages = select_passages_mmr(
                data["question"], data["passages"],
                cross_encoder, sentence_model, topk=args.top_k_snippets,
            )

            for pid, passage in enumerate(passages):
                if len(passage.strip()) < 20:
                    pbar.update(1)
                    continue

                rewrite = get_rewrite(passage, args.model_name,
                                      model, tokenizer, generation_config)

                if rewrite.strip() == passage.strip():
                    print(f"Warning: Rewrite identical to original (pid {pid})")

                qa = get_qa(passage, args.model_name,
                            model, tokenizer, generation_config)

                if not fix_qa(qa)[0]:
                    pbar.update(1)
                    continue

                aug_entry = {
                    "pid": pid,
                    "passage": passage,
                    f"{args.model_name}_rewrite": rewrite,
                    f"{args.model_name}_qa": qa,
                }
                data["augment"].append(aug_entry)
                pbar.update(1)

            # Keep only successfully-augmented passages
            data["passages"] = [a["passage"] for a in data["augment"]]
            ret.append(data)

        pbar.close()
        type_to_ret[qtype] = ret

    # Save datasets
    print("\n### Saving augmented datasets ###")
    save_datasets(type_to_ret, args.model_name)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioASQ augmentation using the golden 13B1 dataset."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model used for rewrite & QA generation (e.g. qwen2.5-7b-instruct)",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the golden BioASQ dataset (data/13B1_golden.json)",
    )
    parser.add_argument(
        "--sample", type=int, default=-1,
        help="Max questions per type (-1 = all)",
    )
    parser.add_argument(
        "--top_k_snippets", type=int, default=3,
        help="Number of top relevant snippets to select per question (default: 3)",
    )

    args = parser.parse_args()
    print(args)
    main(args)
