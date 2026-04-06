"""Tests for Pydantic data models."""
import pytest
from pydantic import ValidationError
from framework.models.chunk import Chapter, Chunk
from framework.models.intake import IntakeForm, AuthorInfo, BookInfo, CompanionConfig
from framework.models.canon_pack import CanonPack


class TestChapterModel:
    def test_valid_chapter(self):
        ch = Chapter(number=1, title="Introduction", raw_text="Some text here.")
        assert ch.number == 1
        assert ch.title == "Introduction"

    def test_chapter_requires_text(self):
        with pytest.raises(ValidationError):
            Chapter(number=1, title="Intro")


class TestChunkModel:
    def test_valid_chunk(self, sample_chunks):
        chunk = sample_chunks[0]
        assert chunk.chapter_number == 1
        assert chunk.position_in_chapter == "beginning"
        assert chunk.is_summary is False

    def test_chunk_preserves_chapter_title(self, sample_chunks):
        assert sample_chunks[0].chapter_title == "The Nature of Conflict"
        assert sample_chunks[1].chapter_title == "Waging War"


class TestIntakeForm:
    def test_valid_intake(self, sample_intake):
        assert sample_intake.book.title == "The Art of War"
        assert len(sample_intake.companion_config.voice_adjectives) == 3

    def test_intake_rejects_invalid_genre(self):
        with pytest.raises(ValidationError):
            IntakeForm(
                author=AuthorInfo(name="Test"),
                book=BookInfo(
                    title="Test Book",
                    genre="invalid_genre",
                    audience="A sufficiently long audience description here",
                    one_sentence="A sufficiently long one-sentence summary here.",
                    publication_status="published",
                ),
                companion_config=CompanionConfig(
                    core_intent="x" * 50,
                    common_misreadings="y" * 30,
                    voice_adjectives=["a", "b", "c"],
                    spoiler_policy="full_discussion",
                    companion_mode="analytical",
                    formality="formal",
                ),
            )

    def test_intake_rejects_short_core_intent(self):
        with pytest.raises(ValidationError):
            CompanionConfig(
                core_intent="too short",
                common_misreadings="y" * 30,
                voice_adjectives=["a", "b", "c"],
                spoiler_policy="full_discussion",
                companion_mode="analytical",
                formality="formal",
            )

    def test_unanswered_questions_coerces_string(self):
        cc = CompanionConfig(
            core_intent="x" * 50,
            common_misreadings="y" * 30,
            voice_adjectives=["a", "b", "c"],
            unanswered_questions="a single question",
            spoiler_policy="full_discussion",
            companion_mode="analytical",
            formality="formal",
        )
        assert cc.unanswered_questions == ["a single question"]

    def test_unanswered_questions_coerces_none(self):
        cc = CompanionConfig(
            core_intent="x" * 50,
            common_misreadings="y" * 30,
            voice_adjectives=["a", "b", "c"],
            unanswered_questions=None,
            spoiler_policy="full_discussion",
            companion_mode="analytical",
            formality="formal",
        )
        assert cc.unanswered_questions == []


class TestCanonPack:
    def test_load_existing_canon_pack(self):
        import json
        from pathlib import Path
        canon_dir = Path("data/canon_packs")
        canon_files = list(canon_dir.glob("*_canon.json"))
        assert len(canon_files) > 0, "No canon pack files found"

        with open(canon_files[0], encoding="utf-8") as f:
            data = json.load(f)
        canon = CanonPack(**data)
        assert canon.metadata.book_title is not None
        assert len(canon.interpretive_framework.chapter_intents) > 0
