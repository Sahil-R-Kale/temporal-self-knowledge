import json, re
import ast
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core.llm_interface import Prompt, create_and_send_prompt


# =========================================================
# CONFIG
# =========================================================

MODEL = "claude-haiku-4-5"
# examples:
# MODEL = "gpt-4.1"
# MODEL = "gpt-4o-mini"

JUDGE_MODEL = "gpt-4.1-mini"

INPUT_PARQUET = "math_qa.parquet"

PRE_OUTPUT_CSV = "outputs/claude_math_pre_conf.csv"
POST_OUTPUT_CSV = "outputs/claude_math_post_conf.csv"

SAVE_EVERY = 2
MAX_RETRIES = 3


SYSTEM_PROMPT = """
You are a careful mathematical QA system.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON.
- Do NOT include markdown.
- Do NOT use ```json fences.
- Do NOT provide explanations outside JSON.
- Do NOT show reasoning unless explicitly requested in a JSON field.
- Your entire response must begin with { and end with }.
- Any response that is not valid JSON is incorrect.

Confidence must reflect calibrated epistemic uncertainty.
"""

# =========================================================
# PROMPTS
# =========================================================

PRE_CONFIDENCE_PROMPT = """
You will be given a mathematical question.

Estimate ONLY your likelihood of answering correctly.

DO NOT:
- solve the problem
- derive equations
- compute intermediate steps
- reveal chain-of-thought
- provide the answer

Return ONLY valid JSON.

Required format:
{{
  "predicted_confidence": float_between_0_and_1,
  "confidence_reasoning": "brief high-level explanation"
}}

Question:
{question}
"""


ANSWER_PROMPT = """
Solve the mathematical question.

Return only the final answer.

Do not use JSON.
Do not explain.
Do not use markdown fences.

Question:
{question}
"""

POST_CONFIDENCE_PROMPT = """
You already answered the following mathematical question.

Question:
{question}

Your answer:
{answer}

Now evaluate the likelihood that your answer is mathematically correct.

Consider:
- symbolic manipulation uncertainty
- algebra/calculus mistakes
- ambiguity
- edge cases
- notation issues
- equivalent forms

Return JSON:

{{
  "final_confidence": float_between_0_and_1,
  "confidence_reasoning": "short explanation"
}}
"""


# =========================================================
# GPT-AS-JUDGE PROMPT
# =========================================================

JUDGE_SYSTEM_PROMPT = """
You are an expert mathematician evaluating whether a generated answer
is mathematically equivalent to the reference answer.

Rules:
- Be strict.
- Mathematical equivalence matters more than formatting.
- Equivalent algebraic forms should be marked correct.
- Ignore formatting differences.
- Ignore extra explanation if the final answer is correct.
- Return valid JSON only.
"""


JUDGE_PROMPT = """
Evaluate whether the model answer is mathematically correct.

Question:
{question}

Ground truth answer:
{ground_truth}

Model answer:
{model_answer}

Return JSON:

{{
  "is_correct": true_or_false,
  "judge_reasoning": "short explanation"
}}
"""


# =========================================================
# JSON PARSING
# =========================================================

def extract_json_block(text: str):

    text = text.strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")

    # try direct parse first
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except:
        pass

    # regex for probable JSON object
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
        f"Could not extract valid JSON from:\n{text}"
    )

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

            return response.content.strip()

        except Exception as e:

            print(
                f"Retry {attempt + 1} failed: {e}"
            )

            time.sleep(2)

    return ""

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

            return extract_json_block(response.content)

        except Exception as e:

            print(
                f"Retry {attempt + 1} failed: {e}"
            )

            time.sleep(2)

    return None


# =========================================================
# GPT-AS-JUDGE EVALUATION
# =========================================================

def evaluate_prediction(
    question,
    ground_truth,
    model_answer,
):

    judge_prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )

    judge_json = generate_json(
        prompt_text=judge_prompt,
        model=JUDGE_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )

    if not judge_json:

        return {
            "is_correct": False,
            "judge_reasoning":
                "Judge evaluation failed.",
            "verdict": "hallucinated",
        }
    
    if not isinstance(judge_json, dict):
        return {
            "is_correct": 0,
            "judge_reasoning":
                "Judge returned invalid format.",
            "verdict": "hallucinated",
        }

    is_correct = judge_json.get(
        "is_correct",
        False,
    )

    judge_reasoning = judge_json.get(
        "judge_reasoning",
        "",
    )

    verdict = (
        "accurate"
        if is_correct
        else "hallucinated"
    )

    return {
        "is_correct": int(is_correct),
        "judge_reasoning": judge_reasoning,
        "verdict": verdict,
    }


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
        total=300,
    ):

        question = row["question"]

        context = (
            row["context"]
            if "context" in row
            else ""
        )

        theorem = (
            row["theorem"]
            if "theorem" in row
            else ""
        )

        link = (
            row["link"]
            if "link" in row
            else ""
        )

        ground_truth = row["answer"]

        sample_id = (
            row["id"]
            if "id" in row
            else idx
        )

        full_question = question

        if isinstance(context, str) and context.strip():

            full_question = f"""
Context:
{context}

Question:
{question}
""".strip()

        # =================================================
        # PRE-CONFIDENCE
        # =================================================

        pre_prompt = (
            PRE_CONFIDENCE_PROMPT.format(
                question=full_question
            )
        )

        pre_json = generate_json(
            prompt_text=pre_prompt,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )

        if not pre_json:
            continue

        predicted_confidence = pre_json.get(
            "predicted_confidence",
            None
        )

        pre_reasoning = pre_json.get(
            "confidence_reasoning",
            ""
        )

        # =================================================
        # ANSWER
        # =================================================

        answer_prompt = (
            ANSWER_PROMPT.format(
                question=full_question
            )
        )

        model_answer = generate_raw(
            prompt_text=answer_prompt,
            model=MODEL,
            system_prompt=(
                "You are a mathematical QA system. "
                "Return only the final answer."
            ),
        )

        if not model_answer:
            continue

        # =================================================
        # POST-CONFIDENCE
        # =================================================

        post_prompt = (
            POST_CONFIDENCE_PROMPT.format(
                question=full_question,
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

        final_confidence = post_json.get(
            "final_confidence",
            None
        )

        post_reasoning = post_json.get(
            "confidence_reasoning",
            ""
        )

        # =================================================
        # GPT-AS-JUDGE EVAL
        # =================================================

        metrics = evaluate_prediction(
            question=question,
            ground_truth=ground_truth,
            model_answer=model_answer,
        )

        # =================================================
        # STORE
        # =================================================

        pre_results.append({

            "row_id": idx,
            "sample_id": sample_id,

            "question":
                question,

            "context":
                context,

            "theorem":
                theorem,

            "link":
                link,

            "predicted_confidence":
                predicted_confidence,

            "confidence_reasoning":
                pre_reasoning,
        })

        post_results.append({

            "row_id": idx,
            "sample_id": sample_id,

            "question":
                question,

            "context":
                context,

            "theorem":
                theorem,

            "link":
                link,

            "ground_truth_answer":
                ground_truth,

            "model_answer":
                model_answer,

            "final_confidence":
                final_confidence,

            "confidence_reasoning":
                post_reasoning,

            "judge_correct":
                metrics["is_correct"],

            "judge_reasoning":
                metrics["judge_reasoning"],

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
                f"Saved progress at row {idx}"
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