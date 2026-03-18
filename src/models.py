from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class StructuredAnswer(BaseModel):
    answer: str = Field(..., description="Answer grounded in the retrieved context only.")
    key_points: List[str] = Field(default_factory=list, description="Main takeaways from the sources.")
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Questions the user might want to ask next.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="How well the retrieved context covers the question."
    )
