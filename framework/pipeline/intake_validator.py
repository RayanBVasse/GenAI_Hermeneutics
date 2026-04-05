"""
Quality gates for intake form validation.
Rule-based checks first, AI-assisted checks for borderline cases.
"""

from dataclasses import dataclass, field

import anthropic

from framework.config import ANTHROPIC_API_KEY


GENERIC_INTENTS = [
    "help readers understand",
    "explain the book",
    "answer questions about",
    "discuss the themes",
    "provide information about",
    "help people learn",
]

GENERIC_ADJECTIVES = [
    "interesting", "good", "nice", "great", "smart", "helpful",
    "informative", "engaging", "compelling", "clear", "simple", "easy",
]


@dataclass
class QualityIssue:
    field: str
    issue: str  # "too_short", "too_generic", "wrong_count", "empty_critical"
    message: str  # Author-facing message
    ai_alternative: str | None = None


def validate_intake(intake: dict) -> list[QualityIssue]:
    """Run all rule-based quality gates. Returns list of issues."""
    issues: list[QualityIssue] = []

    cc = intake.get("companion_config", {})
    book = intake.get("book", {})

    # core_intent
    ci = cc.get("core_intent", "")
    if len(ci) < 50:
        issues.append(QualityIssue(
            "companion_config.core_intent", "too_short",
            "Too short. What specific intellectual work should the companion do?",
        ))
    elif any(g in ci.lower() for g in GENERIC_INTENTS):
        issues.append(QualityIssue(
            "companion_config.core_intent", "too_generic",
            "This could describe any book's companion. What's unique to yours?",
        ))

    # common_misreadings
    cm = cc.get("common_misreadings", "")
    if len(cm) < 30:
        issues.append(QualityIssue(
            "companion_config.common_misreadings", "too_short",
            "What's the most common way readers misinterpret your argument?",
        ))

    # voice_adjectives
    va = cc.get("voice_adjectives", [])
    if len(va) != 3:
        issues.append(QualityIssue(
            "companion_config.voice_adjectives", "wrong_count",
            "Please provide exactly 3 adjectives.",
        ))
    elif len(set(va)) != len(va):
        issues.append(QualityIssue(
            "companion_config.voice_adjectives", "duplicates",
            "Adjectives must be distinct.",
        ))
    else:
        generic = [a for a in va if a.lower() in GENERIC_ADJECTIVES]
        if generic:
            issues.append(QualityIssue(
                "companion_config.voice_adjectives", "too_generic",
                f"'{generic[0]}' is too vague. How does your prose *sound*?",
            ))

    # audience
    aud = book.get("audience", "")
    if len(aud) < 20:
        issues.append(QualityIssue(
            "book.audience", "too_short",
            "Who specifically reads this book?",
        ))

    # one_sentence
    os_val = book.get("one_sentence", "")
    if len(os_val) < 20:
        issues.append(QualityIssue(
            "book.one_sentence", "too_short",
            "One sentence capturing your book's core contribution.",
        ))

    # off_limits — soft prompt, not a blocker
    ol = cc.get("off_limits", [])
    if not ol:
        issues.append(QualityIssue(
            "companion_config.off_limits", "empty_critical",
            "Are you sure? Most authors have at least one topic the companion should avoid.",
        ))

    return issues


def ai_validate_field(
    field_name: str,
    author_answer: str,
    book_title: str,
    author_name: str,
    genre: str,
    one_sentence: str,
) -> QualityIssue | None:
    """
    AI-assisted validation for borderline fields.
    Returns a QualityIssue with suggested alternative, or None if acceptable.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.2,
        messages=[{
            "role": "user",
            "content": f"""The author provided this answer for {field_name}:
"{author_answer}"

The book is: {book_title} by {author_name}
Genre: {genre}
One-sentence summary: {one_sentence}

Is this answer:
1. SPECIFIC enough to differentiate this book from others in the same genre?
2. ACTIONABLE for generating an AI companion Canon Pack?

Respond with JSON:
{{"acceptable": true/false, "suggestion": "improved version if not acceptable", "reason": "brief explanation"}}
Return ONLY valid JSON.""",
        }],
    )

    import json
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    data = json.loads(raw)
    if data.get("acceptable", True):
        return None

    return QualityIssue(
        field=field_name,
        issue="ai_flagged",
        message=data.get("reason", "This answer could be more specific."),
        ai_alternative=data.get("suggestion"),
    )
