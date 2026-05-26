import os

from root_dir_path import ROOT_DIR

current_dataset = None
fewshot = None
fewshot_path = os.path.join(ROOT_DIR, "src", "fewshot")

# ---- Task-specific instruction templates (BioASQ) ----
# Templates for ICL / Combine (passages are provided as SNIPPETS)
TASK_TEMPLATES_WITH_SNIPPETS = {
    "yesno": "Given only the following SNIPPETS and QUESTION, answer the QUESTION only with 'Yes' or 'No'.",
    "factoid": (
        "Extract key biomedical entities strictly using the provided SNIPPETS to answer the QUESTION. "
        "Answer with exactly one highly relevant biomedical entity. Do not provide a list. Just the entity. "
        "If no relevant entities exist, return 'None.'."
    ),
    "list": (
        "Extract key biomedical entities strictly using the provided SNIPPETS to answer the QUESTION. "
        "Return your answer as a strict comma-separated list of entities (e.g. Entity1, Entity2, Entity3). "
        "Do not include any other text. If no relevant entities exist, return 'None'."
    ),
    "summary": (
        "Answer the QUESTION by returning a single paragraph sized text (use max 50 words) ideally "
        "summarizing only the most relevant information in the SNIPPETS."
    ),
}

# Templates for PRAG (no passages – the model relies on question + LoRA weights)
TASK_TEMPLATES_WITHOUT_SNIPPETS = {
    "yesno": "Given only the QUESTION, answer the QUESTION only with 'Yes' or 'No'.",
    "factoid": (
        "Extract key biomedical entities to answer the QUESTION. "
        "Answer with exactly one highly relevant biomedical entity. Do not provide a list. Just the entity. "
        "If no relevant entities exist, return 'None.'."
    ),
    "list": (
        "Extract key biomedical entities to answer the QUESTION. "
        "Return your answer as a strict comma-separated list of entities (e.g. Entity1, Entity2, Entity3). "
        "Do not include any other text. If no relevant entities exist, return 'None'."
    ),
    "summary": "Answer the QUESTION by returning a single paragraph sized text (use max 50 words).",
}


def get_task_instruction(question_type, inference_method):
    """Return the task-specific instruction string, or None if the type is unknown."""
    question_type = question_type.lower()
    if inference_method == "prag":
        return TASK_TEMPLATES_WITHOUT_SNIPPETS.get(question_type)
    else:  # icl / combine
        return TASK_TEMPLATES_WITH_SNIPPETS.get(question_type)


USER_PROMPT = (
    "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"
)

USER_PROMPT_WITH_COT = (
    "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"
)

ASSISTANT_PROMPT = "The answer is {answer}"
ASSISTANT_PROMPT_WITH_COT = "Answer: {answer}"


def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith("?"):
        question = question.strip() + "?"
    elif question.endswith(" ?"):
        question = (question[:-1]).strip() + "?"

    if passages and not isinstance(passages, list):
        passages = [passages]

    if answer is None:
        answer = ""
    else:
        answer = answer.strip()
        if not answer.endswith("."):
            answer += "."
    return question, passages, answer


def get_fewshot(dataset):
    import json

    global current_dataset
    global fewshot
    # assert current_dataset is None
    if dataset.endswith("_golden"):
        dataset = dataset.split("_golden")[0]
    current_dataset = dataset
    with open(os.path.join(fewshot_path, dataset + ".json"), "r") as fin:
        tmp = json.load(fin)
    fewshot = ""
    for data in tmp:
        q = data["question"]
        a = data["answer"]
        fewshot += f"Question: {q}\nAnswer: {a}\n\n"


def get_prompt(
    tokenizer,
    question,
    passages=None,
    answer=None,
    with_cot=False,
    task_instruction=None,
):
    question, passages, answer = _get_prompt(question, passages, answer)
    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid + 1}: {psg}\n"
    if not with_cot:
        user_content = USER_PROMPT.format(question=question, passages=contexts)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT.format(
            question=question, passages=contexts, fewshot=fewshot
        )
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    # Prepend task-specific instruction when provided
    if task_instruction:
        user_content = task_instruction + "\n\n" + user_content

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(inputs, dict) and "input_ids" in inputs:
        inputs = inputs["input_ids"]
    elif hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    if hasattr(inputs, "tolist"):
        inputs = inputs.tolist()
    if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], list):
        inputs = inputs[0]
    inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs
