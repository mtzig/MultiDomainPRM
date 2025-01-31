import time
from typing import Any, Optional

import pandas as pd  # type: ignore
import wandb
from datasets import load_dataset  # type: ignore
from prm_models.search_interface import Search


def make_prompts(data: dict[str, Any]) -> str:
    prompt = f"Subject: {data['subject']}\n"
    prompt += f"Question: {data['question']}\n"
    for i, choice in enumerate(data["choices"]):
        prompt += f"{chr(ord('A') + i)}) {choice}\n"

    return prompt


def ans2idx(ans: str) -> int:
    return ord(ans.upper().strip()[0]) - ord("A")


def mmlu_eval(
    search: Search,
    data_path: str = "prm800k/math_splits/test.jsonl",
    max_n_test: Optional[int] = None,
    name: str = "math_eval",
) -> pd.DataFrame:
    """Evaluates search algorithm on the mmlu dataset. Returns results as `DataFrame`."""
    run = wandb.init(
        dir="results",
        project="strawberry",
        name=name,
    )

    data = load_dataset("cais/mmlu", "all")["test"]  # type: ignore
    if max_n_test is not None:
        data = data[:max_n_test]

    results = []
    n_correct = 0
    n_answered = 0
    n_total = 0

    print("Starting Eval")
    for i, x in enumerate(data):
        start = time.time()
        search_answer = search(x["problem"])
        end = time.time()
        print(f"search_time={end - start:.2f}, {search_answer=}", end=", ")

        if search_answer is None or search_answer == "<answer>":
            grade = False
            search_answer = None
        else:
            grade = grade_answer(search_answer, x["answer"])

        print(f"{grade=}")

        n_answered += search_answer is not None
        n_correct += grade
        n_total += 1

        del x["problem"]
        del x["solution"]
        x["search_answer"] = search_answer if search_answer is not None else pd.NA
        x["grade"] = grade
        results.append(x)

        answer_rate = n_answered / n_total
        accuracy = n_correct / n_total
        print(f"[{i + 1}/{len(data)}] {accuracy=:.2f}, {answer_rate=:.2f}")
        wandb.log({"answer_rate": accuracy, "accuracy": answer_rate}, step=i)

    run.finish()
    return pd.DataFrame(results)


def old(
    search: Search,
    batch_size: int,
) -> pd.DataFrame:
    """Evaluates search algorithm on the mmlu dataset. Returns results as `DataFrame`."""
    data = load_dataset("cais/mmlu", "all")["test"]  # type
    batched_data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    results = []
    for batch in batched_data:
        prompts = make_prompts(batch)
        answers = batch["answer"]
        search_answers = search(prompts)
        grades = [
            ans2idx(a_pred) == a if a_pred is not None else False
            for a_pred, a in zip(search_answers, answers)
        ]
        for s, a_pred, a, g, q in zip(
            batch["subject"], search_answers, answers, grades, batch["question"]
        ):
            results.append(
                {
                    "subject": s,
                    "answer": a,
                    "search_answer": a_pred if a_pred is not None else pd.NA,
                    "grade": g,
                    "question_hash": hash(q),
                }
            )

    return pd.DataFrame(results)
