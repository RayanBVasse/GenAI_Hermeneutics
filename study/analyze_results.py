"""
Results Analyzer.

Reads unblinded evaluation results and computes aggregate statistics:
  - Per-dimension mean scores for each condition
  - Win/loss/tie counts overall and by question category
  - Effect sizes (Cohen's d) per dimension
  - Summary table suitable for a paper

Usage:
  python -m study.analyze_results --book-slug the_fourth_culture_identity_without_borders
  python -m study.analyze_results --all   # analyze all books with results
"""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.config import DATA_DIR


DIMENSIONS = [
    "textual_grounding",
    "interpretive_depth",
    "voice_consistency",
    "boundary_respect",
    "cross_reference",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Compute Cohen's d effect size (positive = group_b better)."""
    if not group_a or not group_b:
        return 0.0
    mean_a, mean_b = _mean(group_a), _mean(group_b)
    std_a, std_b = _std(group_a), _std(group_b)
    pooled_std = math.sqrt((std_a**2 + std_b**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_b - mean_a) / pooled_std


def analyze_book(book_slug: str) -> dict:
    """Analyze unblinded evaluation results for one book."""
    ub_path = DATA_DIR / "results" / book_slug / "evaluation_unblinded.json"
    if not ub_path.exists():
        raise FileNotFoundError(
            f"No unblinded evaluation found: {ub_path}\n"
            f"Run study.evaluate first."
        )

    data = json.loads(ub_path.read_text(encoding="utf-8"))
    evaluations = data["evaluations"]

    # Collect scores by condition and dimension
    scores = {"vanilla_rag": {d: [] for d in DIMENSIONS}, "canon_pack": {d: [] for d in DIMENSIONS}}
    wins = {"vanilla_rag": 0, "canon_pack": 0, "tie": 0}
    category_wins = {}

    for ev in evaluations:
        if "error" in ev:
            continue

        # Win tally
        winner = ev.get("winner_condition", "tie")
        wins[winner] = wins.get(winner, 0) + 1

        cat = ev.get("category", "general")
        if cat not in category_wins:
            category_wins[cat] = {"vanilla_rag": 0, "canon_pack": 0, "tie": 0}
        category_wins[cat][winner] = category_wins[cat].get(winner, 0) + 1

        # Dimension scores
        for condition, dim_scores in ev.get("scores", {}).items():
            if condition in scores:
                for dim in DIMENSIONS:
                    val = dim_scores.get(dim)
                    if val is not None:
                        scores[condition][dim].append(val)

    # Compute aggregates
    summary = {
        "book_slug": book_slug,
        "book_title": data.get("book_title", book_slug),
        "num_evaluated": len([e for e in evaluations if "error" not in e]),
        "wins": wins,
        "category_wins": category_wins,
        "dimensions": {},
    }

    for dim in DIMENSIONS:
        v_scores = scores["vanilla_rag"][dim]
        c_scores = scores["canon_pack"][dim]
        summary["dimensions"][dim] = {
            "vanilla_rag": {"mean": round(_mean(v_scores), 2), "std": round(_std(v_scores), 2), "n": len(v_scores)},
            "canon_pack": {"mean": round(_mean(c_scores), 2), "std": round(_std(c_scores), 2), "n": len(c_scores)},
            "cohens_d": round(_cohens_d(v_scores, c_scores), 3),
            "delta": round(_mean(c_scores) - _mean(v_scores), 2),
        }

    # Overall composite
    all_v = [s for dim in DIMENSIONS for s in scores["vanilla_rag"][dim]]
    all_c = [s for dim in DIMENSIONS for s in scores["canon_pack"][dim]]
    summary["composite"] = {
        "vanilla_rag_mean": round(_mean(all_v), 2),
        "canon_pack_mean": round(_mean(all_c), 2),
        "delta": round(_mean(all_c) - _mean(all_v), 2),
        "cohens_d": round(_cohens_d(all_v, all_c), 3),
    }

    return summary


def print_summary(summary: dict) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: {summary['book_title']}")
    print(f"  Questions evaluated: {summary['num_evaluated']}")
    print(f"{'='*70}")

    # Win tally
    w = summary["wins"]
    print(f"\n  WIN/LOSS/TIE:")
    print(f"    Canon Pack wins: {w.get('canon_pack', 0)}")
    print(f"    Vanilla RAG wins: {w.get('vanilla_rag', 0)}")
    print(f"    Ties:             {w.get('tie', 0)}")

    # Category breakdown
    if summary["category_wins"]:
        print(f"\n  BY CATEGORY:")
        for cat, cw in sorted(summary["category_wins"].items()):
            cp = cw.get("canon_pack", 0)
            vr = cw.get("vanilla_rag", 0)
            ti = cw.get("tie", 0)
            print(f"    {cat:20s}  Canon={cp}  Vanilla={vr}  Tie={ti}")

    # Dimension table
    print(f"\n  {'DIMENSION':<25s} {'VANILLA':>8s} {'CANON':>8s} {'DELTA':>7s} {'d':>7s}")
    print(f"  {'-'*55}")
    for dim in DIMENSIONS:
        d = summary["dimensions"][dim]
        v = d["vanilla_rag"]["mean"]
        c = d["canon_pack"]["mean"]
        delta = d["delta"]
        cd = d["cohens_d"]
        direction = "+" if delta > 0 else ""
        print(f"  {dim:<25s} {v:>8.2f} {c:>8.2f} {direction}{delta:>6.2f} {cd:>7.3f}")

    comp = summary["composite"]
    print(f"  {'-'*55}")
    d_dir = "+" if comp["delta"] > 0 else ""
    print(f"  {'COMPOSITE':<25s} {comp['vanilla_rag_mean']:>8.2f} {comp['canon_pack_mean']:>8.2f} "
          f"{d_dir}{comp['delta']:>6.2f} {comp['cohens_d']:>7.3f}")
    print(f"{'='*70}\n")


def save_summary(summary: dict, book_slug: str) -> Path:
    """Save analysis summary to JSON."""
    out_path = DATA_DIR / "results" / book_slug / "analysis_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Analyze comparison study results.")
    parser.add_argument("--book-slug", help="Book slug to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all books with results")
    args = parser.parse_args()

    if args.all:
        results_dir = DATA_DIR / "results"
        all_summaries = []
        for book_dir in sorted(results_dir.iterdir()):
            ub_path = book_dir / "evaluation_unblinded.json"
            if ub_path.exists():
                summary = analyze_book(book_dir.name)
                print_summary(summary)
                out = save_summary(summary, book_dir.name)
                all_summaries.append(summary)
                print(f"  Saved -> {out}\n")

        if all_summaries:
            _print_cross_book_summary(all_summaries)
        else:
            print("No unblinded evaluation results found. Run evaluate.py first.")
        return

    if not args.book_slug:
        parser.error("Provide --book-slug or --all")

    summary = analyze_book(args.book_slug)
    print_summary(summary)
    out = save_summary(summary, args.book_slug)
    print(f"  Saved -> {out}")


def _print_cross_book_summary(summaries: list[dict]) -> None:
    """Print aggregate results across all books."""
    print(f"\n{'#'*70}")
    print(f"  CROSS-BOOK SUMMARY ({len(summaries)} books)")
    print(f"{'#'*70}")

    total_wins = {"vanilla_rag": 0, "canon_pack": 0, "tie": 0}
    dim_deltas = {d: [] for d in DIMENSIONS}
    dim_effects = {d: [] for d in DIMENSIONS}

    for s in summaries:
        for k in total_wins:
            total_wins[k] += s["wins"].get(k, 0)
        for dim in DIMENSIONS:
            dim_deltas[dim].append(s["dimensions"][dim]["delta"])
            dim_effects[dim].append(s["dimensions"][dim]["cohens_d"])

    print(f"\n  TOTAL WINS: Canon={total_wins['canon_pack']}  "
          f"Vanilla={total_wins['vanilla_rag']}  Tie={total_wins['tie']}")

    print(f"\n  {'DIMENSION':<25s} {'AVG DELTA':>10s} {'AVG d':>8s}")
    print(f"  {'-'*45}")
    for dim in DIMENSIONS:
        avg_d = _mean(dim_deltas[dim])
        avg_e = _mean(dim_effects[dim])
        d_dir = "+" if avg_d > 0 else ""
        print(f"  {dim:<25s} {d_dir}{avg_d:>9.2f} {avg_e:>8.3f}")

    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
