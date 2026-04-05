"""
Blinded A/B Evaluator.

Reads comparison results and uses an AI judge (Claude) to evaluate
each response pair across multiple dimensions. The judge does NOT
know which response is Vanilla RAG vs Canon Pack.

Usage:
  python -m study.evaluate --book-slug the_fourth_culture_identity_without_borders

Dimensions scored (1-5 each):
  - textual_grounding: How well is the answer grounded in the book's actual content?
  - interpretive_depth: Does the answer go beyond surface summary to engage with meaning?
  - voice_consistency: Does the response maintain a coherent, appropriate tone?
  - boundary_respect: Does the companion appropriately handle out-of-scope questions?
  - cross_reference: Does the answer connect ideas across different parts of the book?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.config import ANTHROPIC_API_KEY, DATA_DIR


JUDGE_MODEL = "claude-sonnet-4-20250514"
JUDGE_TEMPERATURE = 0.1  # Low temperature for consistent evaluation
JUDGE_MAX_TOKENS = 2048

EVAL_DIMENSIONS = [
    "textual_grounding",
    "interpretive_depth",
    "voice_consistency",
    "boundary_respect",
    "cross_reference",
]

JUDGE_SYSTEM_PROMPT = """\
You are an expert literary evaluator assessing AI book companion responses.
You will see a reader's question about a book and two companion responses (A and B).
You do NOT know which system generated which response.

Score EACH response on these dimensions (1-5 scale):

1. textual_grounding (1-5): How well is the answer grounded in the book's actual content?
   1=fabricates content, 3=somewhat grounded, 5=precisely anchored in the text

2. interpretive_depth (1-5): Does the answer engage with meaning beyond surface summary?
   1=shallow paraphrase, 3=moderate analysis, 5=rich interpretive engagement

3. voice_consistency (1-5): Does the response maintain a coherent, appropriate companion tone?
   1=generic/robotic, 3=adequate tone, 5=distinctive and well-calibrated voice

4. boundary_respect (1-5): Does the companion handle scope appropriately?
   1=speculates wildly beyond the text, 3=mostly stays in scope, 5=exemplary scope awareness

5. cross_reference (1-5): Does the answer connect ideas across different parts of the book?
   1=no cross-referencing, 3=some connections, 5=rich thematic linking

Also provide:
- winner: "A", "B", or "tie"
- reasoning: 2-3 sentence explanation of why one response is better (or why they tie)

Return ONLY valid JSON in this format:
{
  "response_A": {"textual_grounding": N, "interpretive_depth": N, "voice_consistency": N, "boundary_respect": N, "cross_reference": N},
  "response_B": {"textual_grounding": N, "interpretive_depth": N, "voice_consistency": N, "boundary_respect": N, "cross_reference": N},
  "winner": "A" or "B" or "tie",
  "reasoning": "..."
}"""


def evaluate_pair(
    client: anthropic.Anthropic,
    question: str,
    category: str,
    response_a: str,
    response_b: str,
) -> dict:
    """Evaluate a single A/B pair using the AI judge."""
    user_prompt = (
        f"QUESTION CATEGORY: {category}\n\n"
        f"READER QUESTION:\n{question}\n\n"
        f"RESPONSE A:\n{response_a}\n\n"
        f"RESPONSE B:\n{response_b}\n\n"
        f"Evaluate both responses. Return ONLY valid JSON."
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(raw)


def run_evaluation(book_slug: str) -> dict:
    """Run blinded evaluation on all comparison results for a book."""
    results_path = DATA_DIR / "results" / book_slug / "comparison_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"No comparison results found: {results_path}\n"
            f"Run study.run_comparison first."
        )

    results = json.loads(results_path.read_text(encoding="utf-8"))
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"\n{'='*60}")
    print(f"  EVALUATION: {results['book_title']}")
    print(f"  Comparisons: {len(results['comparisons'])}")
    print(f"  Judge: {JUDGE_MODEL} (temp={JUDGE_TEMPERATURE})")
    print(f"{'='*60}\n")

    evaluations = []
    for comp in results["comparisons"]:
        qid = comp["question_id"]
        print(f"  [{qid}/{len(results['comparisons'])}] Evaluating: {comp['question'][:60]}...")

        t0 = time.time()
        try:
            eval_result = evaluate_pair(
                client,
                question=comp["question"],
                category=comp["category"],
                response_a=comp["response_A"],
                response_b=comp["response_B"],
            )
            eval_result["question_id"] = qid
            eval_result["question"] = comp["question"]
            eval_result["category"] = comp["category"]
            evaluations.append(eval_result)
            print(f"    Winner: {eval_result['winner']} ({time.time()-t0:.1f}s)")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    ERROR: Judge returned invalid JSON: {e}")
            evaluations.append({
                "question_id": qid,
                "question": comp["question"],
                "category": comp["category"],
                "error": str(e),
            })

    # Save evaluations (still blinded)
    eval_output = {
        "book_title": results["book_title"],
        "book_slug": book_slug,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "evaluations": evaluations,
    }

    eval_path = DATA_DIR / "results" / book_slug / "evaluation_blinded.json"
    eval_path.write_text(json.dumps(eval_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Blinded evaluations saved -> {eval_path}")

    # Now unblind and create the decoded results
    key_path = DATA_DIR / "results" / book_slug / "_condition_key.json"
    if key_path.exists():
        key = json.loads(key_path.read_text(encoding="utf-8"))
        unblinded = unblind_evaluations(evaluations, key)
        unblinded_output = {
            **eval_output,
            "evaluations": unblinded,
            "unblinded": True,
        }
        ub_path = DATA_DIR / "results" / book_slug / "evaluation_unblinded.json"
        ub_path.write_text(json.dumps(unblinded_output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Unblinded evaluations saved -> {ub_path}")

    # Print summary
    _print_summary(evaluations)
    print(f"{'='*60}\n")

    return eval_output


def unblind_evaluations(evaluations: list[dict], key: dict) -> list[dict]:
    """Replace A/B labels with actual condition names."""
    unblinded = []
    for ev in evaluations:
        if "error" in ev:
            unblinded.append(ev)
            continue

        qid = str(ev["question_id"])
        label_map = key.get(qid, {})

        entry = {
            "question_id": ev["question_id"],
            "question": ev["question"],
            "category": ev["category"],
            "scores": {},
            "winner_condition": None,
            "reasoning": ev.get("reasoning", ""),
        }

        for label in ["A", "B"]:
            condition = label_map.get(label, f"unknown_{label}")
            scores_key = f"response_{label}"
            if scores_key in ev:
                entry["scores"][condition] = ev[scores_key]

        if ev.get("winner") in ("A", "B"):
            entry["winner_condition"] = label_map.get(ev["winner"], "unknown")
        else:
            entry["winner_condition"] = "tie"

        unblinded.append(entry)

    return unblinded


def _print_summary(evaluations: list[dict]) -> None:
    """Print a quick win/loss tally."""
    wins = {"A": 0, "B": 0, "tie": 0}
    for ev in evaluations:
        if "error" in ev:
            continue
        w = ev.get("winner", "tie")
        wins[w] = wins.get(w, 0) + 1

    print(f"\n  BLINDED TALLY (before unblinding):")
    print(f"    A wins: {wins.get('A', 0)}")
    print(f"    B wins: {wins.get('B', 0)}")
    print(f"    Ties:   {wins.get('tie', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Run blinded A/B evaluation.")
    parser.add_argument("--book-slug", required=True, help="Book slug")
    args = parser.parse_args()

    run_evaluation(args.book_slug)


if __name__ == "__main__":
    main()
