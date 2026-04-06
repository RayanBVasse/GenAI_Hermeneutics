# The Canon Pack: Structured Authorial Intent for GenAI Literary Companions

A research framework for operationalizing authorial interpretive intent in AI-mediated reading experiences.

## Overview

Standard Retrieval-Augmented Generation (RAG) treats books as flat text corpora, retrieving relevant passages but ignoring the author's intended meaning, voice, and interpretive boundaries. This project introduces the **Canon Pack** — a structured JSON representation of authorial intent that shapes how an AI companion interprets and discusses a book.

The system implements a five-stage pipeline:

1. **Parse** — Extract chapters from PDF, DOCX, or TXT manuscripts
2. **Chunk** — Sentence-aware splitting that respects chapter boundaries (512 tokens, 64 overlap)
3. **Embed** — Vector embeddings via OpenAI text-embedding-3-small (1536-dim)
4. **Generate Canon Pack** — AI-assisted extraction of thesis, chapter intents, voice config, boundary rules, and reader guidance
5. **Build System Prompt** — Render the Canon Pack into a companion system prompt

The evaluation study compares a Canon Pack-guided companion against a vanilla RAG baseline across 5 books and 5 interpretive dimensions.

## Repository Structure

```
framework/            Core pipeline (parser, chunker, embedder, canon generator, prompt builder)
framework/models/     Pydantic data models (CanonPack, IntakeForm, Chunk, Chapter)
study/                Evaluation infrastructure (companions, comparison runner, AI judge, analysis)
study/companions/     VanillaRAG and CanonPack companion implementations
study/questions/      Question sets (10 per book, 5 categories)
data/books/           Source texts (not distributed — see data/books/README.md)
data/canon_packs/     Generated Canon Pack JSON files
data/vector_store/    Embedded chunk vectors (numpy-based, JSON storage)
data/results/         Per-book comparison and evaluation results
demo/api/             FastAPI backend for the interactive demo
docs/                 GitHub Pages frontend for the interactive demo
tests/                Unit tests (pytest)
```

## Requirements

- Python 3.10+
- OpenAI API key (embeddings)
- Anthropic API key (Canon Pack generation + companion responses)

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

## Running the Pipeline

```bash
# Process a single book end-to-end
python -m study.run_pipeline \
  --book data/books/fc_iwb.docx \
  --title "The Fourth Culture: Identity Without Borders" \
  --author "Rayan Vasse" \
  --intake data/intake_forms/fc_iwb_intake.json

# Let AI draft the intake form automatically
python -m study.run_pipeline \
  --book data/books/meditations_marcus_aurelius.txt \
  --title "Meditations" \
  --author "Marcus Aurelius"

# Run A/B comparison for a processed book
python -m study.run_comparison \
  --book-slug meditations \
  --author-slug marcus_aurelius \
  --title "Meditations" \
  --author "Marcus Aurelius"

# Evaluate comparison results (blinded AI judge)
python -m study.evaluate --book-slug meditations

# Analyze results and compute effect sizes
python -m study.analyze_results --all
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Technical Decisions

- **Chapter-aware chunking**: Chunks never cross chapter boundaries, preserving structural context
- **Namespace isolation**: Each book's vectors are stored in a separate namespace, preventing cross-contamination
- **Numpy-based vector store**: Custom cosine-similarity retrieval using numpy + JSON (avoids ChromaDB compatibility issues on Python 3.14)
- **Controlled comparison**: Both companions use identical retrieval, model, and temperature — only the system prompt differs
- **Blinded evaluation**: A/B labels are randomized per question; condition keys stored separately and only revealed after scoring

## Evaluation Dimensions

| Dimension | Description |
|-----------|-------------|
| Textual Grounding | How well the response is anchored in actual book content |
| Interpretive Depth | Engagement beyond surface-level paraphrase |
| Voice Consistency | Book-specific tone and character |
| Boundary Respect | Staying within the book's scope, honesty about limits |
| Cross-Reference | Connecting ideas across chapters and themes |

## Cross-Book Results (5 books, 50 questions)

Canon Pack wins 44/50 (88%). Strongest effects on interpretive depth (Cohen's d = 1.89) and voice consistency (d = 1.88).

## Citation

[To be added upon publication]

## License

MIT License. See [LICENSE](LICENSE) for details.
