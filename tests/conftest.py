import pytest
from framework.models.chunk import Chapter, Chunk
from framework.models.intake import (
    IntakeForm, AuthorInfo, BookInfo, CompanionConfig, ChapterSummaries,
)


@pytest.fixture
def sample_chapters():
    """Two short chapters for pipeline testing."""
    return [
        Chapter(
            number=1,
            title="The Nature of Conflict",
            raw_text="War is a matter of vital importance to the State. " * 50,
        ),
        Chapter(
            number=2,
            title="Waging War",
            raw_text="In the operations of war, there are considerations of cost. " * 50,
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Pre-built chunks from sample chapters."""
    return [
        Chunk(
            text="War is a matter of vital importance to the State. " * 10,
            chapter_number=1,
            chapter_title="The Nature of Conflict",
            position_in_chapter="beginning",
            chunk_index=0,
            is_summary=False,
        ),
        Chunk(
            text="In the operations of war, there are considerations of cost. " * 10,
            chapter_number=2,
            chapter_title="Waging War",
            position_in_chapter="beginning",
            chunk_index=1,
            is_summary=False,
        ),
    ]


@pytest.fixture
def sample_intake():
    """A minimal valid IntakeForm matching actual Pydantic constraints."""
    return IntakeForm(
        author=AuthorInfo(name="Sun Tzu"),
        book=BookInfo(
            title="The Art of War",
            genre="nonfiction_general",
            audience="General readers interested in strategy and leadership principles",
            one_sentence="A classical treatise on military strategy and tactical thinking across contexts.",
            publication_status="published",
        ),
        companion_config=CompanionConfig(
            core_intent="Help readers understand strategic principles and apply them beyond military contexts to leadership and decision-making",
            common_misreadings="That the book glorifies violence rather than advocating strategic restraint",
            off_limits=["Modern political commentary"],
            voice_adjectives=["measured", "authoritative", "precise"],
            spoiler_policy="full_discussion",
            companion_mode="analytical",
            formality="formal",
        ),
        chapter_summaries=ChapterSummaries(provided=False, chapters=[]),
    )
