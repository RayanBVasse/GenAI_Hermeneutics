"""Tests for companion setup and condition differentiation (no API calls)."""
import json
from pathlib import Path
from study.companions.vanilla_rag import VanillaRAGCompanion
from study.companions.canon_pack import CanonPackCompanion


class TestVanillaPrompt:
    def test_vanilla_prompt_is_generic(self):
        """VanillaRAG prompt should be short and book-agnostic."""
        comp = VanillaRAGCompanion.__new__(VanillaRAGCompanion)
        comp.book_title = "Test Book"
        comp.author_name = "Test Author"
        prompt = comp.get_system_prompt()
        assert "Test Book" in prompt
        assert "Test Author" in prompt
        assert len(prompt) < 500

    def test_vanilla_prompt_mentions_passages(self):
        comp = VanillaRAGCompanion.__new__(VanillaRAGCompanion)
        comp.book_title = "Test Book"
        comp.author_name = "Test Author"
        prompt = comp.get_system_prompt()
        assert "passages" in prompt.lower()


class TestCanonPackPrompt:
    def test_canon_prompts_are_substantial(self):
        """All generated system prompts should be substantial."""
        prompt_dir = Path("data/system_prompts")
        prompts = list(prompt_dir.glob("*.txt"))
        assert len(prompts) > 0
        for p in prompts:
            text = p.read_text(encoding="utf-8")
            assert len(text) > 500, f"System prompt {p.name} is too short ({len(text)} chars)"

    def test_canon_prompt_longer_than_vanilla(self):
        """Canon Pack prompts should be much longer than the vanilla template."""
        vanilla = VanillaRAGCompanion.__new__(VanillaRAGCompanion)
        vanilla.book_title = "Test"
        vanilla.author_name = "Author"
        vanilla_len = len(vanilla.get_system_prompt())

        prompt_dir = Path("data/system_prompts")
        for p in prompt_dir.glob("*.txt"):
            canon_len = len(p.read_text(encoding="utf-8"))
            assert canon_len > vanilla_len * 3, (
                f"{p.name} ({canon_len} chars) should be much longer than vanilla ({vanilla_len} chars)"
            )


class TestConditionKeys:
    def test_condition_keys_exist(self):
        results_dir = Path("data/results")
        key_files = list(results_dir.glob("*/_condition_key.json"))
        assert len(key_files) > 0

    def test_condition_keys_map_to_both_conditions(self):
        results_dir = Path("data/results")
        key_files = list(results_dir.glob("*/_condition_key.json"))
        for kf in key_files:
            with open(kf, encoding="utf-8") as f:
                keys = json.load(f)
            assert len(keys) > 0
            all_conditions = set()
            for qid, mapping in keys.items():
                assert "A" in mapping and "B" in mapping
                all_conditions.add(mapping["A"])
                all_conditions.add(mapping["B"])
            assert "vanilla_rag" in all_conditions
            assert "canon_pack" in all_conditions
