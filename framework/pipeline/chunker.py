"""
Stage 2: Chapter-Aware Chunker
Splits chapters into semantic chunks that never cross chapter boundaries.
Uses LlamaIndex SentenceSplitter for sentence-aware splitting.
"""

from llama_index.core.node_parser import SentenceSplitter

from framework.config import CHUNK_SIZE, CHUNK_OVERLAP
from framework.models.chunk import Chapter, Chunk


def chunk_chapters(
    chapters: list[Chapter],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chapter_summaries: dict[int, str] | None = None,
) -> list[Chunk]:
    """
    Split parsed chapters into chunks.
    Each chunk stays within its chapter boundary.
    Optionally prepends author-provided chapter summaries as summary chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Chunk] = []
    global_index = 0

    for chapter in chapters:
        chapter_chunks: list[Chunk] = []

        # If author provided a summary for this chapter, add it as first chunk
        if chapter_summaries and chapter.number in chapter_summaries:
            summary_text = chapter_summaries[chapter.number]
            if summary_text.strip():
                chapter_chunks.append(Chunk(
                    text=summary_text.strip(),
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    position_in_chapter="beginning",
                    chunk_index=global_index,
                    is_summary=True,
                ))
                global_index += 1

        # Split chapter text into chunks
        text_splits = splitter.split_text(chapter.raw_text)

        for i, text in enumerate(text_splits):
            if not text.strip():
                continue

            # Determine position
            total = len(text_splits)
            if total <= 1:
                position = "beginning"
            elif i == 0:
                position = "beginning"
            elif i == total - 1:
                position = "end"
            else:
                position = "middle"

            chapter_chunks.append(Chunk(
                text=text.strip(),
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                position_in_chapter=position,
                chunk_index=global_index,
                is_summary=False,
            ))
            global_index += 1

        all_chunks.extend(chapter_chunks)

    return all_chunks


def get_high_signal_chunks(
    chunks: list[Chunk],
    per_chapter: int = 3,
) -> dict[int, list[Chunk]]:
    """
    Select high-signal chunks per chapter for Canon Pack generation.
    Heuristic: longest non-summary chunks (tend to contain denser arguments).
    """
    by_chapter: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.is_summary:
            continue
        by_chapter.setdefault(chunk.chapter_number, []).append(chunk)

    result: dict[int, list[Chunk]] = {}
    for ch_num, ch_chunks in by_chapter.items():
        sorted_by_length = sorted(ch_chunks, key=lambda c: len(c.text), reverse=True)
        result[ch_num] = sorted_by_length[:per_chapter]

    return result
