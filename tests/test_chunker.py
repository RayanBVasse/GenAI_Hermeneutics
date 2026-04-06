"""Tests for the chunking pipeline stage."""
from framework.pipeline.chunker import chunk_chapters, get_high_signal_chunks
from framework.models.chunk import Chapter


class TestChunker:
    def test_chunks_produced(self, sample_chapters):
        chunks = chunk_chapters(sample_chapters)
        assert len(chunks) > 0

    def test_chapter_boundaries_respected(self, sample_chapters):
        """No chunk should contain text from two different chapters."""
        chunks = chunk_chapters(sample_chapters)
        ch1_chunks = [c for c in chunks if c.chapter_number == 1]
        ch2_chunks = [c for c in chunks if c.chapter_number == 2]
        assert len(ch1_chunks) > 0
        assert len(ch2_chunks) > 0
        for c in ch1_chunks:
            assert "considerations of cost" not in c.text
        for c in ch2_chunks:
            assert "vital importance" not in c.text

    def test_chunk_position_labels(self, sample_chapters):
        """First chunk should be 'beginning', last should be 'end' (if multiple)."""
        chunks = chunk_chapters(sample_chapters)
        ch1_chunks = [c for c in chunks if c.chapter_number == 1]
        if len(ch1_chunks) > 1:
            assert ch1_chunks[0].position_in_chapter == "beginning"
            assert ch1_chunks[-1].position_in_chapter == "end"

    def test_small_chapter_single_chunk(self):
        """A very short chapter should produce exactly one chunk."""
        tiny = [Chapter(number=1, title="Brief", raw_text="This is a very short chapter.")]
        chunks = chunk_chapters(tiny)
        assert len(chunks) == 1
        assert chunks[0].position_in_chapter == "beginning"

    def test_chunk_indices_are_sequential(self, sample_chapters):
        chunks = chunk_chapters(sample_chapters)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_high_signal_selects_longest(self, sample_chapters):
        chunks = chunk_chapters(sample_chapters)
        high = get_high_signal_chunks(chunks, per_chapter=1)
        for ch_num, selected in high.items():
            assert len(selected) <= 1
            assert all(not c.is_summary for c in selected)
