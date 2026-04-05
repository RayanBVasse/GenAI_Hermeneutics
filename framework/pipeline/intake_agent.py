"""
AI-Assisted Intake Agent.
Reads parsed chapters + high-signal chunks, generates a complete draft IntakeForm.
The author reviews and edits — flipping the workflow from "author writes" to "author reviews."
"""

import json

import anthropic

from framework.config import ANTHROPIC_API_KEY
from framework.models.intake import IntakeForm
from framework.models.intake_draft import IntakeDraft
from framework.models.chunk import Chapter, Chunk
from framework.pipeline.chunker import chunk_chapters, get_high_signal_chunks


INTAKE_AGENT_MODEL = "claude-sonnet-4-20250514"
INTAKE_AGENT_TEMPERATURE = 0.3
INTAKE_AGENT_MAX_TOKENS = 16384

INTAKE_AGENT_SYSTEM_PROMPT = """\
You are an AI Intake Agent for a book companion system.
You have been given a parsed manuscript. Your job is to generate a complete
author intake form by analyzing the text.

CRITICAL RULES:
1. You are inferring the author's INTENT, not just summarizing text.
2. Every answer must be specific to THIS book — no generic filler.
3. Voice adjectives must reflect the actual prose style, not aspirational qualities.
   Analyze sentence structure, vocabulary level, and rhetorical moves.
4. Common misreadings should anticipate real reader confusions, not strawmen.
5. For fields you're uncertain about, give your best answer AND flag low confidence.
6. Chapter summaries must describe each chapter's argumentative PURPOSE,
   not a plot/content summary — what intellectual work does this chapter do?

OUTPUT FORMAT:
Return a JSON object with exactly two top-level keys:
- "intake": a complete IntakeForm object matching the schema
- "confidence": an object mapping field names to confidence scores (0.0-1.0)

Field names for confidence scores:
  genre, audience, one_sentence, core_intent, common_misreadings,
  voice_adjectives, companion_mode, formality, spoiler_policy,
  off_limits, unanswered_questions, chapter_summaries

Return ONLY valid JSON — no markdown fencing, no commentary.
"""


def generate_draft_intake(
    chapters: list[Chapter],
    chunks: list[Chunk],
    book_title: str,
    author_name: str,
    author_email: str = "",
) -> IntakeDraft:
    """
    Generate a complete draft intake form from manuscript analysis.
    Returns IntakeDraft with per-field confidence scores.
    """
    high_signal = get_high_signal_chunks(chunks, per_chapter=3)

    # Build chapter overview (first 200 words per chapter)
    chapter_overview_parts = []
    for ch in chapters:
        words = ch.raw_text.split()
        preview = " ".join(words[:200])
        chapter_overview_parts.append(
            f"## Chapter {ch.number}: {ch.title}\n"
            f"Word count: {len(words)}\n"
            f"Opening: {preview}..."
        )
    chapter_overview = "\n\n".join(chapter_overview_parts)

    # Build high-signal excerpts
    excerpt_parts = []
    for ch_num in sorted(high_signal.keys()):
        for chunk in high_signal[ch_num]:
            excerpt_parts.append(
                f"[Ch {chunk.chapter_number} — {chunk.chapter_title}]\n"
                f"{chunk.text[:600]}"
            )
    excerpts = "\n\n".join(excerpt_parts)

    total_words = sum(len(ch.raw_text.split()) for ch in chapters)

    user_prompt = f"""MANUSCRIPT ANALYSIS
==================
Title: {book_title}
Author: {author_name}
Chapters parsed: {len(chapters)}
Total words: {total_words:,}

CHAPTER OVERVIEW:
{chapter_overview}

HIGH-SIGNAL EXCERPTS:
{excerpts}

TASK:
Generate a complete intake form for this book as JSON.

The "intake" object must include:
- "author": {{"name": "{author_name}", "email": "{author_email or 'author@example.com'}"}}
- "book": {{title, genre, audience, one_sentence, publication_status: "published"}}
- "companion_config": {{core_intent (min 50 chars), common_misreadings (min 30 chars),
  off_limits (array of strings), voice_adjectives (exactly 3), companion_mode, formality,
  spoiler_policy, unanswered_questions}}
- "chapter_summaries": {{"provided": true, "chapters": [{{number, title, summary}}]}}
- "chapter_summaries": provided + array of chapter summaries

Genre options: nonfiction_general, nonfiction_academic, nonfiction_memoir,
  nonfiction_selfhelp, nonfiction_business, fiction_literary, fiction_genre, hybrid
Companion mode options: reflective, analytical, socratic, guide, mixed
Formality options: casual, conversational, formal, academic
Spoiler policy options: no_spoilers, mild_hints, full_discussion

Return the JSON object with "intake" and "confidence" keys. ONLY valid JSON."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=INTAKE_AGENT_MODEL,
        max_tokens=INTAKE_AGENT_MAX_TOKENS,
        temperature=INTAKE_AGENT_TEMPERATURE,
        system=INTAKE_AGENT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fencing if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    data = json.loads(raw)

    intake = IntakeForm(**data["intake"])
    confidence = data.get("confidence", {})

    needs_attention = [
        field for field, score in confidence.items()
        if score < 0.7
    ]

    return IntakeDraft(
        intake=intake,
        confidence=confidence,
        needs_attention=needs_attention,
        status="draft",
    )
