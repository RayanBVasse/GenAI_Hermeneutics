"""
Blinded A/B Comparison Runner.

For each question in a book's question set, runs both companions
(Vanilla RAG vs Canon Pack) and saves blinded results where the
condition labels are randomly assigned to A or B.

Usage:
  python -m study.run_comparison \
    --book-slug the_fourth_culture_identity_without_borders \
    --author-slug rayan_vasse \
    --title "The Fourth Culture: Identity Without Borders" \
    --author "Rayan Vasse"

  # Run all books defined in the question sets directory:
  python -m study.run_comparison --all
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.config import DATA_DIR
from study.companions.vanilla_rag import VanillaRAGCompanion
from study.companions.canon_pack import CanonPackCompanion


def load_questions(book_slug: str) -> list[dict]:
    """Load the question set for a book."""
    q_path = PROJECT_ROOT / "study" / "questions" / f"{book_slug}.json"
    if not q_path.exists():
        raise FileNotFoundError(f"No question set found: {q_path}")
    return json.loads(q_path.read_text(encoding="utf-8"))


def run_comparison(
    book_slug: str,
    author_slug: str,
    book_title: str,
    author_name: str,
    canon_pack_path: str | None = None,
) -> dict:
    """
    Run blinded A/B comparison for one book.
    Returns the full results dict.
    """
    questions = load_questions(book_slug)
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {book_title}")
    print(f"  Questions: {len(questions)}")
    print(f"{'='*60}\n")

    # Initialize both companions
    vanilla = VanillaRAGCompanion(author_slug, book_slug, book_title, author_name)
    canon = CanonPackCompanion(author_slug, book_slug, canon_pack_path)

    results = {
        "book_title": book_title,
        "author_name": author_name,
        "book_slug": book_slug,
        "author_slug": author_slug,
        "num_questions": len(questions),
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.4,
        "comparisons": [],
        # Key mapping is stored but should NOT be revealed during evaluation
        "_condition_key": {},
    }

    for i, q_entry in enumerate(questions, 1):
        question = q_entry["question"]
        category = q_entry.get("category", "general")

        print(f"  [{i}/{len(questions)}] {question[:70]}...")

        # Run both companions
        t0 = time.time()
        vanilla_resp = vanilla.ask(question)
        canon_resp = canon.ask(question)
        elapsed = time.time() - t0

        # Blind the labels: randomly assign A/B
        if random.random() < 0.5:
            label_map = {"A": "vanilla_rag", "B": "canon_pack"}
            response_a = vanilla_resp
            response_b = canon_resp
        else:
            label_map = {"A": "canon_pack", "B": "vanilla_rag"}
            response_a = canon_resp
            response_b = vanilla_resp

        comparison = {
            "question_id": i,
            "question": question,
            "category": category,
            "response_A": response_a.answer,
            "response_B": response_b.answer,
            "retrieved_chunks": [
                {"chapter": c["chapter_number"], "title": c["chapter_title"], "score": c["score"]}
                for c in vanilla_resp.retrieved_chunks
            ],
            "elapsed_seconds": round(elapsed, 1),
        }
        results["comparisons"].append(comparison)
        results["_condition_key"][str(i)] = label_map

        print(f"    Done ({elapsed:.1f}s) | A={len(response_a.answer)} chars, B={len(response_b.answer)} chars")

    # Save results
    out_dir = DATA_DIR / "results" / book_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save key separately (for unblinding after evaluation)
    key_path = out_dir / "_condition_key.json"
    key_path.write_text(json.dumps(results["_condition_key"], indent=2), encoding="utf-8")

    print(f"\n  Results saved -> {out_path}")
    print(f"  Condition key -> {key_path} (do not open until evaluation is complete)")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run blinded A/B comparison study.")
    parser.add_argument("--book-slug", help="Book slug (e.g. the_fourth_culture_identity_without_borders)")
    parser.add_argument("--author-slug", help="Author slug (e.g. rayan_vasse)")
    parser.add_argument("--title", help="Book title")
    parser.add_argument("--author", help="Author name")
    parser.add_argument("--canon-pack", default=None, help="Path to Canon Pack JSON (optional)")
    parser.add_argument("--all", action="store_true", help="Run all books with question sets")

    args = parser.parse_args()

    if args.all:
        q_dir = PROJECT_ROOT / "study" / "questions"
        q_files = sorted(q_dir.glob("*.json"))
        if not q_files:
            print("ERROR: No question set files found in study/questions/")
            sys.exit(1)
        for q_file in q_files:
            data = json.loads(q_file.read_text(encoding="utf-8"))
            meta = data[0].get("_meta", {}) if data and "_meta" in data[0] else {}
            # Question files must include metadata in first entry or be loaded via manifest
            # For now, require explicit args per book
            print(f"Found: {q_file.stem} ({len(data)} questions)")
        print("\nUse --book-slug, --author-slug, --title, --author for each book.")
        return

    if not all([args.book_slug, args.author_slug, args.title, args.author]):
        parser.error("Provide --book-slug, --author-slug, --title, and --author")

    run_comparison(
        book_slug=args.book_slug,
        author_slug=args.author_slug,
        book_title=args.title,
        author_name=args.author,
        canon_pack_path=args.canon_pack,
    )


if __name__ == "__main__":
    main()
