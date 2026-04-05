from pydantic import BaseModel


class Chapter(BaseModel):
    number: int
    title: str
    raw_text: str


class Chunk(BaseModel):
    text: str
    chapter_number: int
    chapter_title: str
    position_in_chapter: str  # "beginning", "middle", "end"
    chunk_index: int
    is_summary: bool = False
