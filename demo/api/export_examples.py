"""
Export pre-computed A/B comparison examples from data/results/ into
compact JSON files suitable for the demo frontend.

Run from repo root:
    python -m demo.api.export_examples
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

BOOKS = {
    "meditations": {
        "title": "Meditations",
        "author": "Marcus Aurelius",
        "author_slug": "marcus_aurelius",
        "year": "c. 170 CE",
        "description": "Private philosophical journal of the Roman Emperor, exploring Stoic ethics, impermanence, and duty.",
    },
    "the_art_of_war": {
        "title": "The Art of War",
        "author": "Sun Tzu",
        "author_slug": "sun_tzu",
        "year": "c. 5th century BCE",
        "description": "Ancient Chinese treatise on military strategy, leadership, and the philosophy of conflict.",
    },
    "content": {
        "title": "Content",
        "author": "Cory Doctorow",
        "author_slug": "cory_doctorow",
        "year": "2008",
        "description": "Essays on digital rights, copyright reform, and the future of creative work in the internet age.",
    },
    "being_no_one": {
        "title": "Being No One",
        "author": "Thomas Metzinger",
        "author_slug": "thomas_metzinger",
        "year": "2003",
        "description": "A theory of subjectivity exploring the self-model theory of consciousness and phenomenal experience.",
    },
    "the_fourth_culture_identity_without_borders": {
        "title": "The Fourth Culture: Identity Without Borders",
        "author": "Rayan B. Vasse",
        "author_slug": "rayan_vasse",
        "year": "2024",
        "description": "An exploration of hybrid identity formation for those living between cultures, proposing a 'fourth culture' framework.",
    },
}


def export_book(slug: str, meta: dict) -> dict | None:
    """Export comparison examples + evaluation for one book."""
    book_dir = RESULTS_DIR / slug

    comp_path = book_dir / "comparison_results.json"
    eval_path = book_dir / "evaluation_unblinded.json"
    analysis_path = book_dir / "analysis_summary.json"

    if not comp_path.exists():
        print(f"  SKIP {slug}: no comparison_results.json")
        return None

    comp = json.loads(comp_path.read_text(encoding="utf-8"))
    evals = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else None

    # Build evaluation lookup by question_id
    eval_lookup = {}
    if evals:
        for ev in evals["evaluations"]:
            eval_lookup[ev["question_id"]] = ev

    # Build compact examples
    examples = []
    for c in comp["comparisons"]:
        qid = c["question_id"]
        ev = eval_lookup.get(qid, {})

        examples.append({
            "question_id": qid,
            "question": c["question"],
            "category": c["category"],
            "vanilla_rag_response": c.get("response_A") if _is_vanilla(comp, c, "A") else c.get("response_B"),
            "canon_pack_response": c.get("response_B") if _is_vanilla(comp, c, "A") else c.get("response_A"),
            "scores": ev.get("scores", {}),
            "winner": ev.get("winner_condition", ""),
            "reasoning": ev.get("reasoning", ""),
        })

    result = {
        "slug": slug,
        **meta,
        "num_questions": len(examples),
        "examples": examples,
    }

    if analysis:
        result["summary"] = {
            "wins": analysis.get("wins", {}),
            "dimensions": analysis.get("dimensions", {}),
            "composite": analysis.get("composite", {}),
        }

    return result


def _is_vanilla(comp: dict, comparison: dict, label: str) -> bool:
    """
    Determine if response_A is vanilla_rag.
    We check the condition key file, or fall back to heuristic (vanilla is shorter).
    """
    book_dir = RESULTS_DIR / comp["book_slug"]
    key_path = book_dir / "_condition_key.json"
    if key_path.exists():
        key = json.loads(key_path.read_text(encoding="utf-8"))
        qid = str(comparison["question_id"])
        mapping = key.get(qid, key.get(comparison["question_id"], {}))
        if mapping:
            return mapping.get("A") == "vanilla_rag"

    # Fallback heuristic: vanilla responses tend to be shorter
    a_len = len(comparison.get("response_A", ""))
    b_len = len(comparison.get("response_B", ""))
    return a_len < b_len


def main():
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    books_index = []
    for slug, meta in BOOKS.items():
        print(f"Exporting {slug}...")
        result = export_book(slug, meta)
        if result:
            out_path = EXAMPLES_DIR / f"{slug}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  -> {out_path.name} ({len(result['examples'])} examples)")
            books_index.append({
                "slug": slug,
                "title": meta["title"],
                "author": meta["author"],
                "year": meta["year"],
                "description": meta["description"],
                "num_questions": len(result["examples"]),
                "canon_wins": result.get("summary", {}).get("wins", {}).get("canon_pack", 0),
                "vanilla_wins": result.get("summary", {}).get("wins", {}).get("vanilla_rag", 0),
            })

    # Write index
    index_path = EXAMPLES_DIR / "_index.json"
    index_path.write_text(json.dumps(books_index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nIndex written with {len(books_index)} books.")


if __name__ == "__main__":
    main()
