from typing import Optional
from pydantic import BaseModel, Field


class ChapterIntent(BaseModel):
    chapter_number: int
    chapter_title: str
    author_intent: str
    key_concepts: list[str] = Field(default_factory=list)
    emotional_arc: str = ""


class CrossReference(BaseModel):
    concept: str
    appears_in_chapters: list[int] = Field(default_factory=list)
    note: str = ""


class InterpretiveFramework(BaseModel):
    book_thesis: str
    chapter_intents: list[ChapterIntent] = Field(default_factory=list)
    foreground_themes: list[str] = Field(default_factory=list)
    background_themes: list[str] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)


class SampleResponse(BaseModel):
    reader_question: str
    ideal_companion_response: str


class PronounRules(BaseModel):
    author_reference: str = "third person: the author"
    reader_reference: str = "second person: you"


class VoiceConfig(BaseModel):
    tone: str
    formality: str
    companion_mode: str
    pronoun_rules: PronounRules = Field(default_factory=PronounRules)
    sample_responses: list[SampleResponse] = Field(default_factory=list)


class BoundaryRules(BaseModel):
    off_limits_topics: list[str] = Field(default_factory=list)
    never_do: list[str] = Field(default_factory=list)
    spoiler_policy: str = "full_discussion"
    fallback_response: str = (
        "That's outside what this companion discusses. "
        "You might want to reach out to the author directly."
    )


class ReaderGuidance(BaseModel):
    common_misreadings: list[str] = Field(default_factory=list)
    suggested_entry_points: list[str] = Field(default_factory=list)
    unanswered_questions: list[str] = Field(default_factory=list)


class RetrievalConfig(BaseModel):
    vector_namespace: str
    chunk_count: int = 0
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    hybrid_search: bool = False


class CanonPackMetadata(BaseModel):
    book_title: str
    author_name: str
    generated_at: str
    pipeline_version: str = "1.0"
    status: str = "draft"
    review_notes: Optional[str] = None


class CanonPack(BaseModel):
    metadata: CanonPackMetadata
    interpretive_framework: InterpretiveFramework
    voice_config: VoiceConfig
    boundary_rules: BoundaryRules
    reader_guidance: ReaderGuidance
    retrieval_config: RetrievalConfig
