# response_classifier.py

import re


def classify_response(query, parametric_ans, contextual_ans, response, judge_model) -> str:
    prompt = f"""Task: Classify RESPONSE based on similarity to two reference answers.
    Rules:
    1. Output ONLY: PARAMETRIC_ANSWER, CONTEXTUAL_ANSWER, or neither.
    2. If RESPONSE supports BOTH, output CONTEXTUAL_ANSWER.
    3. Allow for paraphrasing and synonyms.
    4. Refusals ("I don't know") = neither.

    Input:
    Query: {query}
    PARAMETRIC_ANSWER: {parametric_ans}
    CONTEXTUAL_ANSWER: {contextual_ans}
    RESPONSE: {response}
    IS THE RESPONSE MORE SIMILAR TO PARAMETRIC_ANSWER OR CONTEXTUAL_ANSWER? (RESPOND IN ONE WORD)"""

    raw = judge_model.generate(prompt, temperature=0.0, max_tokens=3) or ""

    # Normalize
    label = raw.strip()
    label = label.split()[0] if label else ""
    label = label.strip("`\"'.,:;()[]{}")

    label_up = label.upper()
    label_up = re.sub(r"[^A-Z_]", "", label_up)

    ALIASES = {
        "PARAMETRIC": "PARAMETRIC_ANSWER",
        "PARAMETRICANSWER": "PARAMETRIC_ANSWER",
        "PARAMETRIC_ANSWER": "PARAMETRIC_ANSWER",
        "CONTEXTUAL": "CONTEXTUAL_ANSWER",
        "CONTEXTUALANSWER": "CONTEXTUAL_ANSWER",
        "CONTEXTUAL_ANSWER": "CONTEXTUAL_ANSWER",
        "NEITHER": "neither",
    }

    mapped = ALIASES.get(label_up, None)
    if mapped is None:
        return raw
    return mapped



def classify_response_factual_recall(query, parametric_ans, response, judge_model) -> str:
    prompt = f"""Task: Classify MODEL_RESPONSE as CORRECT or INCORRECT according to GROUNDTRUTH_ANSWER
    Rules:
    1. Output ONLY: CORRECT or INCORRECT.
    3. Allow for paraphrasing and synonyms.
    4. Refusals ("I don't know") = INCORRECT.

    Input:
    Query: {query}
    GROUNDTRUTH_ANSWER: {parametric_ans}
    MODEL_RESPONSE: {response}
    IS THE MODEL_RESPONSE CORRECT OR INCORRECT (GIVEN THE GROUNDTRUTH_ANSWER)? (RESPOND IN ONE WORD)"""

    raw = judge_model.generate(prompt, temperature=0.0, max_tokens=3) or ""

    # Normalize
    label = raw.strip()
    label = label.split()[0] if label else ""
    label = label.strip("`\"'.,:;()[]{}")

    label_up = label.upper()
    label_up = re.sub(r"[^A-Z_]", "", label_up)

    ALIASES = {
        "PARAMETRIC": "PARAMETRIC_ANSWER",
        "PARAMETRICANSWER": "PARAMETRIC_ANSWER",
        "PARAMETRIC_ANSWER": "PARAMETRIC_ANSWER",
        "CONTEXTUAL": "CONTEXTUAL_ANSWER",
        "CONTEXTUALANSWER": "CONTEXTUAL_ANSWER",
        "CONTEXTUAL_ANSWER": "CONTEXTUAL_ANSWER",
        "NEITHER": "neither",
    }

    mapped = ALIASES.get(label_up, None)
    if mapped is None:
        return raw
    return mapped
