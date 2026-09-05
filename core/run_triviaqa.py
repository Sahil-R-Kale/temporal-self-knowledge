import json
import ast
import time
import string
import re

from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

from core.llm_interface import Prompt, create_and_send_prompt


# =========================================================
# CONFIG
# =========================================================

MODEL = "claude-sonnet-4-6"
# or:
# MODEL = "gpt-4.1"

INPUT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "trivia_qa.parquet"

PRE_OUTPUT_CSV = "outputs/claude_triviaqa_pre_conf.csv"
POST_OUTPUT_CSV = "outputs/claude_triviaqa_post_conf.csv"

SAVE_EVERY = 2
MAX_RETRIES = 3


SYSTEM_PROMPT = """
You are a careful factual QA system.

Requirements:
- Be concise.
- Do not fabricate facts.
- If uncertain, still provide your best answer.
- Confidence must reflect calibrated epistemic uncertainty.
- Confidence is NOT a probability of sounding plausible.
- Confidence is your estimated probability the final answer is factually correct.

Return valid JSON only.
"""


# =========================================================
# PROMPTS
# =========================================================

PRE_CONFIDENCE_PROMPT = """
You will be given a trivia question.

Before answering:
1. Estimate how likely you are to answer correctly.
2. Think about topic familiarity, ambiguity, and retrieval certainty.
3. Do NOT solve the question internally in detail.

Return JSON:

{{
  "predicted_confidence": float_between_0_and_1,
  "confidence_reasoning": "short explanation"
}}

Question:
{question}
"""


ANSWER_PROMPT = """
Answer the trivia question.

Return JSON:

{{
  "answer": "final short answer"
}}

Question:
{question}
"""


POST_CONFIDENCE_PROMPT = """
You already answered the following trivia question.

Question:
{question}

Your answer:
{answer}

Now evaluate the likelihood that your answer is factually correct.

Consider:
- retrieval certainty
- ambiguity
- possible confusion with nearby entities
- historical/date uncertainty
- naming variations

Return JSON:

{{
  "final_confidence": float_between_0_and_1,
  "confidence_reasoning": "short explanation"
}}
"""


# =========================================================
# JSON PARSING
# =========================================================

def safe_json_parse(text: str):

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]

    if text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)

    except:

        try:
            return ast.literal_eval(text)

        except:
            raise ValueError(
                f"Could not parse JSON:\n{text}"
            )


# =========================================================
# OFFICIAL TRIVIAQA EVALUATION
# =========================================================

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):

        exclude = set(
            string.punctuation +
            "".join([u"‘", u"’", u"´", u"`"])
        )

        return ''.join(
            ch if ch not in exclude else ' '
            for ch in text
        )

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    replace_underscore(s)
                )
            )
        )
    ).strip()


def f1_score(prediction, ground_truth):

    prediction_tokens = normalize_answer(
        prediction
    ).split()

    ground_truth_tokens = normalize_answer(
        ground_truth
    ).split()

    common = Counter(prediction_tokens) & Counter(
        ground_truth_tokens
    )

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = (
        1.0 * num_same / len(prediction_tokens)
    )

    recall = (
        1.0 * num_same / len(ground_truth_tokens)
    )

    return (
        2 * precision * recall
    ) / (precision + recall)


def exact_match_score(prediction, ground_truth):

    return (
        normalize_answer(prediction)
        ==
        normalize_answer(ground_truth)
    )


def metric_max_over_ground_truths(
    metric_fn,
    prediction,
    ground_truths
):

    return max([
        metric_fn(prediction, gt)
        for gt in ground_truths
    ])


def get_ground_truths(answer_field):

    if isinstance(answer_field, str):
        answer_field = ast.literal_eval(answer_field)

    aliases = answer_field.get("aliases", [])

    value = answer_field.get("value", "")

    ground_truths = aliases + [value]

    return list(set([
        normalize_answer(x)
        for x in ground_truths
        if x
    ]))


def evaluate_prediction(prediction, answer_field):

    ground_truths = get_ground_truths(answer_field)

    em = metric_max_over_ground_truths(
        exact_match_score,
        prediction,
        ground_truths
    )

    f1 = metric_max_over_ground_truths(
        f1_score,
        prediction,
        ground_truths
    )

    verdict = (
        "accurate"
        if em
        else "hallucinated"
    )

    return {
        "exact_match": int(em),
        "f1": round(f1, 4),
        "verdict": verdict,
        "ground_truths": ground_truths,
    }


# =========================================================
# LLM CALL
# =========================================================

@create_and_send_prompt
def call_llm(user_prompt: str, model: str):

    return Prompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
    )


def generate_json(prompt_text, model=MODEL):

    for attempt in range(MAX_RETRIES):

        try:

            response = call_llm(
                user_prompt=prompt_text,
                model=model,
                temperature=0.0,
            )

            return safe_json_parse(
                response.content
            )

        except Exception as e:

            print(
                f"Retry {attempt + 1} failed: {e}"
            )

            time.sleep(2)

    return None


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
        total=400,
    ):

        question = row["question"]
        question_id = row["question_id"]

        # =================================================
        # PRE-CONFIDENCE
        # =================================================

        pre_prompt = (
            PRE_CONFIDENCE_PROMPT.format(
                question=question
            )
        )

        pre_json = generate_json(
            pre_prompt
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
                question=question
            )
        )

        answer_json = generate_json(
            answer_prompt
        )

        if not answer_json:
            continue

        model_answer = answer_json.get(
            "answer",
            ""
        )

        # =================================================
        # POST-CONFIDENCE
        # =================================================

        post_prompt = (
            POST_CONFIDENCE_PROMPT.format(
                question=question,
                answer=model_answer,
            )
        )

        post_json = generate_json(
            post_prompt
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
        # EVALUATION
        # =================================================

        metrics = evaluate_prediction(
            model_answer,
            row["answer"]
        )

        # =================================================
        # STORE
        # =================================================

        pre_results.append({

            "row_id": idx,
            "question_id": question_id,
            "question": question,


            "predicted_confidence":
                predicted_confidence,

            "confidence_reasoning":
                pre_reasoning,
        })

        post_results.append({

            "row_id": idx,
            "question_id": question_id,
            "question": question,

            "reference_aliases":
                "|".join(
                    metrics["ground_truths"]
                ),

            "model_answer":
                model_answer,

            "final_confidence":
                final_confidence,

            "confidence_reasoning":
                post_reasoning,

            "exact_match":
                metrics["exact_match"],

            "f1":
                metrics["f1"],

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
