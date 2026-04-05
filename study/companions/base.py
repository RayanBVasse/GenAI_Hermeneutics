"""
Base companion interface.
Both VanillaRAG and CanonPack companions share the same retrieval
and generation pipeline -- only the system prompt differs.
"""

from dataclasses import dataclass

import anthropic

from framework.config import ANTHROPIC_API_KEY, RETRIEVAL_TOP_K
from framework.pipeline.embedder import retrieve


# Shared generation settings -- identical across both conditions
COMPANION_MODEL = "claude-sonnet-4-20250514"
COMPANION_TEMPERATURE = 0.4
COMPANION_MAX_TOKENS = 1024


@dataclass
class CompanionResponse:
    question: str
    answer: str
    retrieved_chunks: list[dict]
    system_prompt_used: str
    model: str
    temperature: float
    condition: str  # "vanilla_rag" or "canon_pack"


class BaseCompanion:
    """Abstract base for both study conditions."""

    condition: str = "base"

    def __init__(self, author_slug: str, book_slug: str):
        self.author_slug = author_slug
        self.book_slug = book_slug
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def get_system_prompt(self) -> str:
        raise NotImplementedError

    def ask(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> CompanionResponse:
        """
        Retrieve context and generate a response.
        Retrieval is identical for both conditions.
        """
        # Stage 1: Retrieve (shared)
        chunks = retrieve(
            author_slug=self.author_slug,
            book_slug=self.book_slug,
            query=question,
            top_k=top_k,
        )

        # Stage 2: Build user message with retrieved context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Passage {i} - Chapter {chunk['chapter_number']}: "
                f"{chunk['chapter_title']}]\n{chunk['text']}"
            )
        context_block = "\n\n".join(context_parts)

        user_message = (
            f"RETRIEVED PASSAGES:\n{context_block}\n\n"
            f"READER QUESTION:\n{question}"
        )

        # Stage 3: Generate (same model, temperature, max_tokens)
        system_prompt = self.get_system_prompt()
        response = self.client.messages.create(
            model=COMPANION_MODEL,
            max_tokens=COMPANION_MAX_TOKENS,
            temperature=COMPANION_TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return CompanionResponse(
            question=question,
            answer=response.content[0].text,
            retrieved_chunks=chunks,
            system_prompt_used=system_prompt,
            model=COMPANION_MODEL,
            temperature=COMPANION_TEMPERATURE,
            condition=self.condition,
        )
