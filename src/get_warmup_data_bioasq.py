import os
import json
import random
from collections import defaultdict

from root_dir_path import ROOT_DIR

random.seed(42)

MAX_PER_TYPE = 3000


def get_answer(question):
    """Extract the answer string from a BioASQ question based on its type."""
    qtype = question["type"]

    if qtype == "yesno":
        # exact_answer is "yes" or "no"
        return question["exact_answer"]

    elif qtype == "factoid":
        # exact_answer is a list of acceptable answers, take the first
        answer = question["exact_answer"]
        if isinstance(answer, list):
            return answer[0]
        return answer

    elif qtype == "list":
        # exact_answer is a list of lists (each inner list = aliases for one item)
        # Flatten to a comma-separated string of the first alias per item
        items = question["exact_answer"]
        flat = [item[0] if isinstance(item, list) else item for item in items]
        return ", ".join(flat)

    elif qtype == "summary":
        # No exact_answer, use ideal_answer
        if "ideal_answer" not in question:
            return None
        answer = question["ideal_answer"]
        if isinstance(answer, list):
            return answer[0]
        return answer

    return None


def main():
    input_path = os.path.join(ROOT_DIR, "data", "training14b_clean.json")
    with open(input_path, "r") as fin:
        data = json.load(fin)

    # Group questions by type
    by_type = defaultdict(list)
    for q in data["questions"]:
        by_type[q["type"]].append(q)

    output_dir = os.path.join(ROOT_DIR, "warmup", "data", "direct")
    os.makedirs(output_dir, exist_ok=True)

    for qtype, questions in by_type.items():
        random.shuffle(questions)
        selected = questions[:MAX_PER_TYPE]

        dataset = []
        for q in selected:
            answer = get_answer(q)
            if answer is None or len(answer.strip()) == 0:
                continue
            dataset.append({
                "question": q["body"].strip(),
                "answer": answer.strip(),
            })

        output_path = os.path.join(output_dir, f"bioasq_{qtype}.json")
        with open(output_path, "w") as fout:
            json.dump(dataset, fout, indent=4)

        print(f"{qtype}: {len(dataset)} questions saved to {output_path}")


if __name__ == "__main__":
    main()
