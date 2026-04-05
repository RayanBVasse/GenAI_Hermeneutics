"""
Canon Pack Companion -- experimental condition.
Uses the full Canon Pack-generated system prompt with the same retrieval pipeline.
"""

import json
from pathlib import Path

from framework.config import DATA_DIR
from framework.pipeline.prompt_builder import build_system_prompt
from framework.models.canon_pack import CanonPack
from framework.pipeline.embedder import _slugify
from study.companions.base import BaseCompanion


class CanonPackCompanion(BaseCompanion):

    condition = "canon_pack"

    def __init__(self, author_slug: str, book_slug: str, canon_pack_path: str | Path | None = None):
        super().__init__(author_slug, book_slug)

        # Load Canon Pack from explicit path or default location
        if canon_pack_path:
            cp_path = Path(canon_pack_path)
        else:
            cp_path = DATA_DIR / "canon_packs" / f"{book_slug}_canon.json"

        if not cp_path.exists():
            raise FileNotFoundError(f"Canon Pack not found: {cp_path}")

        data = json.loads(cp_path.read_text(encoding="utf-8"))
        self.canon = CanonPack(**data)
        self._system_prompt = build_system_prompt(self.canon)

    def get_system_prompt(self) -> str:
        return self._system_prompt
