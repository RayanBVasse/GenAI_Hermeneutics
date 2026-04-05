from pydantic import BaseModel, Field
from framework.models.intake import IntakeForm


class IntakeDraft(BaseModel):
    intake: IntakeForm
    confidence: dict[str, float] = Field(default_factory=dict)
    needs_attention: list[str] = Field(default_factory=list)
    status: str = "draft"  # draft → validated → approved
