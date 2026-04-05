"""
CLI Pipeline Runner for Computational Hermeneutic study.

Runs the full Canon Ingestion Pipeline on a single book:
  parse -> chunk -> embed -> (optional AI intake) -> generate Canon Pack -> build system prompt

Usage:
  python -m study.run_pipeline \
    --book data/books/fc_iwb.docx \
    --title "From Chaos I Was Born" \
    --author "Rayan Vasse" \
    --intake data/intake_forms/fc_iwb_intake.json  # optional

If --intake is omitted, the AI Intake Agent generates a draft automatically.
"""

import argparse
import json
import sys
import time
from functools import wraps
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.config import DATA_DIR
from framework.models.intake import IntakeForm
from framework.pipeline.parser import parse_file
from framework.pipeline.chunker import chunk_chapters
from framework.pipeline.embedder import embed_chunks, _slugify
from framework.pipeline.canon_generator import generate_canon_pack
from framework.pipeline.prompt_builder import build_system_prompt


def _log(stage: str, msg: str) -> None:
    print(f"  [{stage}] {msg}")


def _retry_on_rate_limit(func, *args, max_retries=3, **kwargs):
    """Retry a function call if it hits a rate limit error."""
    import anthropic as _anth
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except _anth.RateLimitError:
            wait = 60 * (attempt + 1)
            _log("RETRY", f"Rate limit hit. Waiting {wait}s before retry {attempt+2}/{max_retries}...")
            time.sleep(wait)
    return func(*args, **kwargs)  # final attempt, let it raise


def run_pipeline(
    book_path: str,
    book_title: str,
    author_name: str,
    intake_path: str | None = None,
    author_email: str = "",
    skip_embed: bool = False,
) -> dict:
    """
    Execute the full pipeline and return paths to all generated artifacts.
    """
    results = {}
    book_slug = _slugify(book_title)
    author_slug = _slugify(author_name)
    namespace = f"{author_slug}_{book_slug}"

    print(f"\n{'='*60}")
    print(f"  PIPELINE: {book_title} by {author_name}")
    print(f"  Source:   {book_path}")
    print(f"{'='*60}\n")

    # ── Stage 1: Parse ──────────────────────────────────────────
    t0 = time.time()
    _log("PARSE", f"Parsing {Path(book_path).name}...")
    chapters = parse_file(book_path)
    total_words = sum(len(ch.raw_text.split()) for ch in chapters)
    _log("PARSE", f"Found {len(chapters)} chapters, {total_words:,} words ({time.time()-t0:.1f}s)")
    results["chapters"] = len(chapters)
    results["total_words"] = total_words

    # ── Stage 2: Chunk ──────────────────────────────────────────
    t0 = time.time()
    _log("CHUNK", "Chunking chapters...")

    # If intake has chapter summaries, pass them
    chapter_summaries = None
    if intake_path:
        intake_data = json.loads(Path(intake_path).read_text(encoding="utf-8"))
        intake = IntakeForm(**intake_data)
        if intake.chapter_summaries.provided and intake.chapter_summaries.chapters:
            chapter_summaries = {
                cs.number: cs.summary
                for cs in intake.chapter_summaries.chapters
            }

    chunks = chunk_chapters(chapters, chapter_summaries=chapter_summaries)
    avg_words = sum(len(c.text.split()) for c in chunks) / max(len(chunks), 1)
    _log("CHUNK", f"Created {len(chunks)} chunks, avg {avg_words:.0f} words ({time.time()-t0:.1f}s)")
    results["chunks"] = len(chunks)

    # ── Stage 3: Embed ──────────────────────────────────────────
    if skip_embed:
        _log("EMBED", "Skipping embedding (--skip-embed)")
        results["namespace"] = namespace
    else:
        t0 = time.time()
        _log("EMBED", f"Embedding {len(chunks)} chunks with OpenAI...")
        embed_result = embed_chunks(chunks, author_name, book_title)
        namespace = embed_result["namespace"]
        _log("EMBED", f"Stored {embed_result['chunks_embedded']} vectors -> "
             f"data/vector_store/{namespace}.json ({time.time()-t0:.1f}s)")
        results["namespace"] = namespace

    # ── Stage 3.5: Intake (if not provided) ─────────────────────
    if intake_path:
        _log("INTAKE", f"Using provided intake: {intake_path}")
        intake_data = json.loads(Path(intake_path).read_text(encoding="utf-8"))
        intake = IntakeForm(**intake_data)
    else:
        t0 = time.time()
        _log("INTAKE", "No intake provided -- running AI Intake Agent...")
        from framework.pipeline.intake_agent import generate_draft_intake
        draft = _retry_on_rate_limit(
            generate_draft_intake,
            chapters=chapters,
            chunks=chunks,
            book_title=book_title,
            author_name=author_name,
            author_email=author_email,
        )
        intake = draft.intake

        # Save the AI-generated draft
        draft_path = DATA_DIR / "intake_forms" / f"{book_slug}_ai_draft.json"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(
            json.dumps(draft.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        _log("INTAKE", f"AI draft saved -> {draft_path.name}")

        # Report confidence
        low_conf = [f for f, s in draft.confidence.items() if s < 0.7]
        if low_conf:
            _log("INTAKE", f"Low-confidence fields: {', '.join(low_conf)}")
        _log("INTAKE", f"Done ({time.time()-t0:.1f}s)")
        results["intake_draft_path"] = str(draft_path)

    # ── Stage 4: Generate Canon Pack ────────────────────────────
    t0 = time.time()
    _log("CANON", "Generating Canon Pack with Claude Sonnet...")
    canon = _retry_on_rate_limit(generate_canon_pack, intake, chapters, chunks, namespace)
    _log("CANON", f"Canon Pack: {len(canon.interpretive_framework.chapter_intents)} chapter intents, "
         f"{len(canon.interpretive_framework.cross_references)} cross-refs, "
         f"{len(canon.voice_config.sample_responses)} sample responses ({time.time()-t0:.1f}s)")

    # Save Canon Pack
    canon_path = DATA_DIR / "canon_packs" / f"{book_slug}_canon.json"
    canon_path.parent.mkdir(parents=True, exist_ok=True)
    canon_path.write_text(
        json.dumps(canon.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )
    _log("CANON", f"Saved -> {canon_path.name}")
    results["canon_path"] = str(canon_path)

    # ── Stage 5: Build System Prompt ────────────────────────────
    t0 = time.time()
    _log("PROMPT", "Building companion system prompt...")
    prompt = build_system_prompt(canon)

    prompt_path = DATA_DIR / "system_prompts" / f"{book_slug}_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    _log("PROMPT", f"Saved -> {prompt_path.name} ({len(prompt):,} chars, {time.time()-t0:.1f}s)")
    results["prompt_path"] = str(prompt_path)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Chapters: {results['chapters']}")
    print(f"  Chunks:   {results['chunks']}")
    print(f"  Namespace: {results.get('namespace', 'N/A')}")
    print(f"  Canon Pack: {canon_path}")
    print(f"  System Prompt: {prompt_path}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the Canon Ingestion Pipeline on a book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With pre-written intake form
  python -m study.run_pipeline \\
    --book data/books/fc_iwb.docx \\
    --title "From Chaos I Was Born" \\
    --author "Rayan Vasse" \\
    --intake data/intake_forms/fc_iwb_intake.json

  # Let AI draft the intake
  python -m study.run_pipeline \\
    --book data/books/meditations_marcus_aurelius.txt \\
    --title "Meditations" \\
    --author "Marcus Aurelius"

  # Skip embedding (for testing parse/chunk only)
  python -m study.run_pipeline \\
    --book data/books/art_of_war_sun_tzu.txt \\
    --title "The Art of War" \\
    --author "Sun Tzu" \\
    --skip-embed
""",
    )
    parser.add_argument("--book", required=True, help="Path to manuscript (DOCX, PDF, or TXT)")
    parser.add_argument("--title", required=True, help="Book title")
    parser.add_argument("--author", required=True, help="Author name")
    parser.add_argument("--intake", default=None, help="Path to intake form JSON (optional)")
    parser.add_argument("--email", default="", help="Author email (optional)")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding stage")

    args = parser.parse_args()

    # Resolve paths relative to project root
    book_path = Path(args.book)
    if not book_path.is_absolute():
        book_path = PROJECT_ROOT / book_path
    if not book_path.exists():
        print(f"ERROR: Book file not found: {book_path}")
        sys.exit(1)

    intake_path = None
    if args.intake:
        intake_path = Path(args.intake)
        if not intake_path.is_absolute():
            intake_path = PROJECT_ROOT / intake_path
        if not intake_path.exists():
            print(f"ERROR: Intake form not found: {intake_path}")
            sys.exit(1)
        intake_path = str(intake_path)

    run_pipeline(
        book_path=str(book_path),
        book_title=args.title,
        author_name=args.author,
        intake_path=intake_path,
        author_email=args.email,
        skip_embed=args.skip_embed,
    )


if __name__ == "__main__":
    main()
