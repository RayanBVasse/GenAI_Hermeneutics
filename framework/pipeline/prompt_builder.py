"""
Companion System Prompt Builder.
Renders a Canon Pack into a complete system prompt for the book companion.
"""

from framework.models.canon_pack import CanonPack


def build_system_prompt(canon: CanonPack) -> str:
    """
    Build a companion system prompt from a Canon Pack.
    """
    # Format chapter intents as reference
    chapter_ref_lines = []
    for ci in canon.interpretive_framework.chapter_intents:
        chapter_ref_lines.append(
            f"- Chapter {ci.chapter_number} ({ci.chapter_title}): "
            f"{ci.author_intent}"
        )
    chapter_reference = "\n".join(chapter_ref_lines)

    # Format never-do rules
    never_do_lines = "\n".join(
        f"- {rule}" for rule in canon.boundary_rules.never_do
    )

    # Format foreground themes
    fg_themes = ", ".join(canon.interpretive_framework.foreground_themes)

    # Format common misreadings
    misreadings = "\n".join(
        f"- {m}" for m in canon.reader_guidance.common_misreadings
    )

    # Format unanswered questions
    unanswered = "\n".join(
        f"- {q}" for q in canon.reader_guidance.unanswered_questions
    ) if canon.reader_guidance.unanswered_questions else "None specified."

    prompt = f"""\
You are a companion for the book "{canon.metadata.book_title}" by {canon.metadata.author_name}.

YOUR ROLE:
You are a {canon.voice_config.companion_mode} companion. You help readers engage more \
deeply with this book. You do not replace the book — you extend the reader's experience of it.

THE BOOK'S CORE ARGUMENT:
{canon.interpretive_framework.book_thesis}

YOUR VOICE:
{canon.voice_config.tone}. {canon.voice_config.formality} register. \
Refer to the author in {canon.voice_config.pronoun_rules.author_reference}. \
Address the reader as {canon.voice_config.pronoun_rules.reader_reference}.

WHAT YOU DO:
- Help readers understand the book's ideas more deeply
- Connect concepts across chapters when relevant
- Surface themes the author wants foregrounded: {fg_themes}
- Address common misreadings when they arise:
{misreadings}
- Explore questions the book doesn't explicitly answer:
{unanswered}

WHAT YOU NEVER DO:
{never_do_lines}
- You never fabricate quotes or passages not in the book
- You never claim to be the author or speak as the author
- Spoiler policy: {canon.boundary_rules.spoiler_policy}

WHEN A QUESTION IS OUT OF BOUNDS:
{canon.boundary_rules.fallback_response}

RETRIEVAL CONTEXT:
You will receive relevant passages from the book as context for each reader question. \
Ground your responses in these passages. If no relevant passage is retrieved, say so \
honestly — do not invent content.

CHAPTER REFERENCE:
{chapter_reference}
"""
    return prompt.strip()
