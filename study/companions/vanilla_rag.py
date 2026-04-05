"""
Vanilla RAG Companion -- control condition.
Uses a generic, book-agnostic system prompt with the same retrieval pipeline.
"""

from study.companions.base import BaseCompanion


class VanillaRAGCompanion(BaseCompanion):

    condition = "vanilla_rag"

    def __init__(self, author_slug: str, book_slug: str, book_title: str, author_name: str):
        super().__init__(author_slug, book_slug)
        self.book_title = book_title
        self.author_name = author_name

    def get_system_prompt(self) -> str:
        return (
            f'You are a helpful assistant that answers questions about the book '
            f'"{self.book_title}" by {self.author_name}.\n\n'
            f'You will receive relevant passages from the book as context. '
            f'Use these passages to answer the reader\'s question accurately and thoroughly. '
            f'If the passages do not contain enough information to answer, say so honestly.\n\n'
            f'Be clear, informative, and grounded in the text.'
        )
