import json
import time
import re

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.llm_interface import (
    Prompt,
    create_and_send_prompt,
)


# =========================================================
# CONFIG
# =========================================================

MODEL = "claude-sonnet-4-5"

INPUT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "riddle_bench.parquet"

PRE_OUTPUT_CSV = (
    "outputs/claude_logic_pre_conf.csv"
)

POST_OUTPUT_CSV = (
    "outputs/claude_logic_post_conf.csv"
)

SAVE_EVERY = 2
MAX_RETRIES = 3


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a careful logical reasoning system.

Rules:
- Be concise.
- Return short answers.
- Do not output reasoning unless requested.
- Confidence should reflect true uncertainty.
"""


# =========================================================
# PROMPTS
# =========================================================

PRE_CONFIDENCE_PROMPT = """
You will be given a logic problem.

Estimate likelihood of solving correctly.

Do not solve it at all.

Return JSON of only confidence, no other output tokens will be allowed. The JSON should be in the following format:

{{
  "predicted_confidence": float,
  "confidence_reasoning": "brief reason"
}}

Question:
{question}
"""


ANSWER_PROMPT = """
Solve the logic problem.

Return only the final answer. Absolutely no other output tokens are allowed.

Rules:
- No explanation
- No reasoning
- No JSON
- No markdown
- Keep answer extremely short containing only the final answer, ideally one word or number.

Question:
{question}
"""


POST_CONFIDENCE_PROMPT = """
You answered the following logic problem.

Question:
{question}

Your answer:
{answer}

Estimate probability your answer is correct. Return only the final confidence in the given format. Absolutely no other output tokens are allowed.

Return JSON only:

{{
  "final_confidence": float,
  "confidence_reasoning": "brief reason"
}}
"""


# =========================================================
# JSON PARSING
# =========================================================

def extract_json_block(text: str):

    text = text.strip()

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except:
        pass

    candidates = re.findall(
        r'\{[\s\S]*\}',
        text
    )

    for candidate in reversed(candidates):

        try:

            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except:
            continue

    raise ValueError(
        f"Could not extract JSON:\n{text}"
    )


# =========================================================
# VALIDATION
# =========================================================

def validate_confidence_json(
    obj,
    key,
):

    if not isinstance(obj, dict):
        return None

    if key not in obj:
        return None

    try:

        value = float(obj[key])

    except:
        return None

    value = max(
        0.0,
        min(1.0, value)
    )

    obj[key] = value

    return obj


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(text):

    text = str(text).strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =========================================================
# EVALUATION
# =========================================================

def evaluate_prediction(
    prediction,
    ground_truth,
):

    pred = normalize(prediction)

    gt = normalize(ground_truth)

    correct = int(pred == gt)

    return {

        "exact_match": correct,

        "verdict": (
            "accurate"
            if correct
            else "hallucinated"
        ),
    }


# =========================================================
# LLM CALL
# =========================================================

@create_and_send_prompt
def call_llm(
    user_prompt: str,
    model: str,
    system_prompt: str,
):

    return Prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
    )


# =========================================================
# JSON GENERATION
# =========================================================

def generate_json(
    prompt_text,
    model,
    system_prompt,
):

    for attempt in range(MAX_RETRIES):

        try:

            response = call_llm(
                user_prompt=prompt_text,
                model=model,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            return extract_json_block(
                response.content
            )

        except Exception as e:

            print(
                f"Retry {attempt + 1} "
                f"failed: {e}"
            )

            time.sleep(2)

    return None


# =========================================================
# RAW GENERATION
# =========================================================

def generate_raw(
    prompt_text,
    model,
    system_prompt,
):

    for attempt in range(MAX_RETRIES):

        try:

            response = call_llm(
                user_prompt=prompt_text,
                model=model,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            answer = (
                response.content
                .strip()
                .split("\n")[0]
                .strip()
            )

            return answer

        except Exception as e:

            print(
                f"Retry {attempt + 1} "
                f"failed: {e}"
            )

            time.sleep(2)

    return ""


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline():

    Path("outputs").mkdir(
        exist_ok=True
    )

    df = pd.read_parquet(
        INPUT_PARQUET
    )

    pre_results = []
    post_results = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
    ):

        question = row["question"]

        ground_truth = str(
            row["answer"]
        )

        sample_id = (
            row["id"]
            if "id" in row
            else idx
        )

        task_type = (
            row["type"]
            if "type" in row
            else ""
        )

        # =================================================
        # PRE CONFIDENCE
        # =================================================

        pre_prompt = (
            PRE_CONFIDENCE_PROMPT.format(
                question=question
            )
        )

        pre_json = generate_json(
            prompt_text=pre_prompt,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )

        if not pre_json:
            continue

        pre_json = (
            validate_confidence_json(
                pre_json,
                "predicted_confidence",
            )
        )

        if not pre_json:
            continue

        predicted_confidence = (
            pre_json.get(
                "predicted_confidence"
            )
        )

        pre_reasoning = (
            pre_json.get(
                "confidence_reasoning",
                ""
            )
        )

        # =================================================
        # ANSWER
        # =================================================

        answer_prompt = (
            ANSWER_PROMPT.format(
                question=question
            )
        )

        model_answer = generate_raw(
            prompt_text=answer_prompt,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )

        if not model_answer:
            continue

        # =================================================
        # POST CONFIDENCE
        # =================================================

        post_prompt = (
            POST_CONFIDENCE_PROMPT.format(
                question=question,
                answer=model_answer,
            )
        )

        post_json = generate_json(
            prompt_text=post_prompt,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )

        if not post_json:
            continue

        post_json = (
            validate_confidence_json(
                post_json,
                "final_confidence",
            )
        )

        if not post_json:
            continue

        final_confidence = (
            post_json.get(
                "final_confidence"
            )
        )

        post_reasoning = (
            post_json.get(
                "confidence_reasoning",
                ""
            )
        )

        # =================================================
        # EVALUATION
        # =================================================

        metrics = evaluate_prediction(
            prediction=model_answer,
            ground_truth=ground_truth,
        )

        # =================================================
        # STORE PRE
        # =================================================

        pre_results.append({

            "row_id": idx,

            "sample_id":
                sample_id,

            "task_type":
                task_type,

            "question":
                question,

            "predicted_confidence":
                predicted_confidence,

            "confidence_reasoning":
                pre_reasoning,
        })

        # =================================================
        # STORE POST
        # =================================================

        post_results.append({

            "row_id": idx,

            "sample_id":
                sample_id,

            "task_type":
                task_type,

            "question":
                question,

            "ground_truth_answer":
                ground_truth,

            "model_answer":
                model_answer,

            "final_confidence":
                final_confidence,

            "confidence_reasoning":
                post_reasoning,

            "exact_match":
                metrics["exact_match"],

            "verdict":
                metrics["verdict"],
        })

        # =================================================
        # SAVE
        # =================================================

        if (
            idx % SAVE_EVERY == 0
            and idx > 0
        ):

            pd.DataFrame(
                pre_results
            ).to_csv(
                PRE_OUTPUT_CSV,
                index=False,
            )

            pd.DataFrame(
                post_results
            ).to_csv(
                POST_OUTPUT_CSV,
                index=False,
            )

            print(
                f"Saved progress "
                f"at row {idx}"
            )

    # =====================================================
    # FINAL SAVE
    # =====================================================

    pd.DataFrame(
        pre_results
    ).to_csv(
        PRE_OUTPUT_CSV,
        index=False,
    )

    pd.DataFrame(
        post_results
    ).to_csv(
        POST_OUTPUT_CSV,
        index=False,
    )

    print("Done.")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run_pipeline()
