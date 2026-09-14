"""Strict transport schema for stateless anonymous public conversations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnonymousChatRequest(BaseModel):
    """Accept text only; identity and client-supplied capabilities are forbidden."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=2000)


class AnonymousChatResponse(BaseModel):
    """Return public dialogue without user, group, or profile state."""

    request_id: str
    reply: str
    can_write_profile: Literal[False] = False
