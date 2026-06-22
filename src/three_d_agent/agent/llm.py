"""LLM client factory. Swap providers via env or config."""
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-3-5-sonnet-latest"
    temperature: float = 0.0
    max_tokens: int = 4096


def build_llm(cfg: LLMConfig) -> BaseChatModel:
    if cfg.provider == "anthropic":
        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if cfg.provider == "openai":
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    raise ValueError(f"Unknown provider: {cfg.provider}")
