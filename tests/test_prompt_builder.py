"""Tests for system prompt generation from Canon Pack."""
import json
from pathlib import Path
from framework.models.canon_pack import CanonPack
from framework.pipeline.prompt_builder import build_system_prompt


class TestPromptBuilder:
    def _load_canon(self):
        canon_dir = Path("data/canon_packs")
        canon_files = list(canon_dir.glob("*_canon.json"))
        assert len(canon_files) > 0
        with open(canon_files[0], encoding="utf-8") as f:
            data = json.load(f)
        return CanonPack(**data)

    def test_prompt_is_string(self):
        canon = self._load_canon()
        prompt = build_system_prompt(canon)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_contains_book_title(self):
        canon = self._load_canon()
        prompt = build_system_prompt(canon)
        assert canon.metadata.book_title.lower() in prompt.lower()

    def test_prompt_contains_boundary_rules(self):
        canon = self._load_canon()
        prompt = build_system_prompt(canon)
        assert any(
            term in prompt.lower()
            for term in ["never", "do not", "off-limits", "out of bounds"]
        )

    def test_prompt_contains_voice_config(self):
        canon = self._load_canon()
        prompt = build_system_prompt(canon)
        assert canon.voice_config.formality.lower() in prompt.lower()

    def test_prompt_contains_chapter_intents(self):
        canon = self._load_canon()
        prompt = build_system_prompt(canon)
        assert "chapter" in prompt.lower()
        # At least one chapter title should appear
        first_intent = canon.interpretive_framework.chapter_intents[0]
        assert first_intent.chapter_title in prompt
