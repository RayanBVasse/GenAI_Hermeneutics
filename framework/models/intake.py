from typing import Optional
from pydantic import BaseModel, Field


class AuthorInfo(BaseModel):
    name: str
    email: str = ""
    website: Optional[str] = None


class BookInfo(BaseModel):
    title: str
    subtitle: Optional[str] = None
    genre: str = Field(
        ...,
        pattern=r"^(nonfiction_general|nonfiction_academic|nonfiction_memoir|nonfiction_selfhelp|nonfiction_business|fiction_literary|fiction_genre|hybrid)$",
    )
    audience: str = Field(..., min_length=20)
    one_sentence: str = Field(..., min_length=20)
    publication_status: str = Field(
        ..., pattern=r"^(published|in_progress|planned)$"
    )
    isbn: Optional[str] = None
    amazon_url: Optional[str] = None
    word_count_approx: Optional[int] = None


class CompanionConfig(BaseModel):
    core_intent: str = Field(..., min_length=50)
    common_misreadings: str = Field(..., min_length=30)
    off_limits: list[str] = Field(default_factory=list)
    voice_adjectives: list[str] = Field(..., min_length=3, max_length=3)
    unanswered_questions: Optional[str] = None
    spoiler_policy: str = Field(
        ..., pattern=r"^(no_spoilers|mild_hints|full_discussion)$"
    )
    companion_mode: str = Field(
        ..., pattern=r"^(reflective|analytical|socratic|guide|mixed)$"
    )
    formality: str = Field(
        ..., pattern=r"^(casual|conversational|formal|academic)$"
    )


class ChapterSummaryEntry(BaseModel):
    number: int
    title: str
    summary: str


class ChapterSummaries(BaseModel):
    provided: bool = False
    chapters: list[ChapterSummaryEntry] = Field(default_factory=list)


class IntakeForm(BaseModel):
    author: AuthorInfo
    book: BookInfo
    companion_config: CompanionConfig
    chapter_summaries: ChapterSummaries = Field(
        default_factory=ChapterSummaries
    )
