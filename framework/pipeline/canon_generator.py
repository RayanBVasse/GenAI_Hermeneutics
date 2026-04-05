"""
Stage 4: Canon Pack Generator
Uses Claude Sonnet to generate a structured Canon Pack from
the author's intake form + manuscript chunks.
"""

import json
from datetime import datetime, timezone

import anthropic

from framework.config import (
    ANTHROPIC_API_KEY,
    CANON_MODEL,
    CANON_TEMPERATURE,
    CANON_MAX_TOKENS,
)
from framework.models.intake import IntakeForm
from framework.models.chunk import Chapter, Chunk
from framework.models.canon_pack import CanonPack
from framework.pipeline.chunker import get_high_signal_chunks


CANON_SYSTEM_PROMPT = """\
You are generating a Canon Pack for a book companion.

A Canon Pack defines how an AI companion should interpret, discuss, and guide readers \
through a specific book. It captures what the author MEANS, not just what the text SAYS.

Given the author's intake form answers and the book's chapter content, produce a \
structured Canon Pack in JSON format.

CRITICAL RULES:
- Be specific to THIS book. Name specific concepts, characters, arguments, and frameworks.
- The book_thesis must capture the core argument in 1-2 sentences — not a generic summary.
- Each chapter_intent must describe what the author MEANS this chapter to do for the reader.
- foreground_themes are ideas the companion should actively surface in conversation.
- background_themes are present in the book but should not be foregrounded unprompted.
- cross_references track concepts that evolve across chapters.
- sample_responses must be in the configured voice — they are the quality benchmark.
- never_do rules must include the defaults plus any book-specific restrictions.
- suggested_entry_points are good first questions a new reader might ask.
"""


def _build_canon_prompt(
    intake: IntakeForm,
    chapters: list[Chapter],
    high_signal_chunks: dict[int, list[Chunk]],
) -> str:
    """Build the user prompt for Canon Pack generation."""

    # Chapter summaries section
    chapter_info = []
    for ch in chapters:
        summary_lines = [f"Chapter {ch.number}: {ch.title}"]
        # Add high-signal chunk excerpts
        if ch.number in high_signal_chunks:
            for chunk in high_signal_chunks[ch.number]:
                excerpt = chunk.text[:300].strip()
                summary_lines.append(f"  Excerpt: {excerpt}")
        chapter_info.append("\n".join(summary_lines))

    chapters_text = "\n\n".join(chapter_info)

    return f"""\
## AUTHOR INTAKE FORM

**Book Title:** {intake.book.title}
**Author:** {intake.author.name}
**Genre:** {intake.book.genre}
**Audience:** {intake.book.audience}
**One-Sentence Summary:** {intake.book.one_sentence}

**Core Intent:** {intake.companion_config.core_intent}
**Common Misreadings:** {intake.companion_config.common_misreadings}
**Off-Limits Topics:** {", ".join(intake.companion_config.off_limits) if intake.companion_config.off_limits else "None specified"}
**Voice Adjectives:** {", ".join(intake.companion_config.voice_adjectives)}
**Companion Mode:** {intake.companion_config.companion_mode}
**Formality:** {intake.companion_config.formality}
**Spoiler Policy:** {intake.companion_config.spoiler_policy}
{"**Unanswered Questions:** " + intake.companion_config.unanswered_questions if intake.companion_config.unanswered_questions else ""}

## CHAPTER CONTENT (high-signal excerpts)

{chapters_text}

## INSTRUCTIONS

Generate a complete Canon Pack JSON matching this exact structure:

{{
  "metadata": {{
    "book_title": "{intake.book.title}",
    "author_name": "{intake.author.name}",
    "generated_at": "{datetime.now(timezone.utc).isoformat()}",
    "pipeline_version": "1.0",
    "status": "draft",
    "review_notes": null
  }},
  "interpretive_framework": {{
    "book_thesis": "1-2 sentence core argument/thesis",
    "chapter_intents": [
      {{
        "chapter_number": <number>,
        "chapter_title": "<title>",
        "author_intent": "<what the author means this chapter to do for the reader>",
        "key_concepts": ["<named frameworks, terms, ideas>"],
        "emotional_arc": "<how the reader should feel by the end>"
      }}
    ],
    "foreground_themes": ["<themes to actively surface>"],
    "background_themes": ["<themes present but not to foreground>"],
    "cross_references": [
      {{
        "concept": "<concept name>",
        "appears_in_chapters": [<chapter numbers>],
        "note": "<how the concept evolves>"
      }}
    ]
  }},
  "voice_config": {{
    "tone": "{', '.join(intake.companion_config.voice_adjectives)}",
    "formality": "{intake.companion_config.formality}",
    "companion_mode": "{intake.companion_config.companion_mode}",
    "pronoun_rules": {{
      "author_reference": "third person: the author",
      "reader_reference": "second person: you"
    }},
    "sample_responses": [
      {{
        "reader_question": "<plausible reader question>",
        "ideal_companion_response": "<50-150 words, in the configured voice>"
      }}
    ]
  }},
  "boundary_rules": {{
    "off_limits_topics": {json.dumps(intake.companion_config.off_limits)},
    "never_do": [
      "Never speculate about the author's personal life",
      "Never provide clinical or therapeutic interpretations",
      "Never summarise the entire book in one response",
      "Never fabricate quotes or passages not in the book",
      "Never claim to be the author or speak as the author"
    ],
    "spoiler_policy": "{intake.companion_config.spoiler_policy}",
    "fallback_response": "That's outside what this companion discusses. You might want to reach out to the author directly."
  }},
  "reader_guidance": {{
    "common_misreadings": ["<expanded from intake>"],
    "suggested_entry_points": ["<3-5 good first questions>"],
    "unanswered_questions": ["<from intake, expanded>"]
  }},
  "retrieval_config": {{
    "vector_namespace": "<will be filled by pipeline>",
    "chunk_count": 0,
    "embedding_model": "text-embedding-3-small",
    "top_k": 5,
    "hybrid_search": false
  }}
}}

Generate chapter_intents for ALL {len(chapters)} chapters.
Generate at least 3 sample_responses.
Generate at least 3 cross_references.
Generate 3-5 suggested_entry_points.
Return ONLY valid JSON — no markdown fencing, no commentary.
"""


def generate_canon_pack(
    intake: IntakeForm,
    chapters: list[Chapter],
    chunks: list[Chunk],
    namespace: str,
) -> CanonPack:
    """
    Generate a Canon Pack using Claude Sonnet.
    """
    high_signal = get_high_signal_chunks(chunks, per_chapter=3)
    prompt = _build_canon_prompt(intake, chapters, high_signal)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=CANON_MODEL,
        max_tokens=CANON_MAX_TOKENS,
        temperature=CANON_TEMPERATURE,
        system=CANON_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown fencing if present
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)

    canon_data = json.loads(raw_text)

    # Patch retrieval config with actual values
    canon_data["retrieval_config"]["vector_namespace"] = namespace
    canon_data["retrieval_config"]["chunk_count"] = len(chunks)

    return CanonPack(**canon_data)
