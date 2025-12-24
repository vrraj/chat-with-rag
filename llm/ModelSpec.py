# backend/llm/modelspec.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal

Provider = Literal["openai", "gemini", "anthropic"]

@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str

    # Common knobs (optional defaults)
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None

    # Provider-specific extras (tooling, safety settings, etc.)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> Dict[str, Any]:
        """Flatten spec into kwargs suitable for handler.create(...).

        Precedence: `extra` (defaults) < explicit fields (temperature/max_output_tokens).
        This prevents `extra` from accidentally overriding core, provider-agnostic knobs.
        """
        out: Dict[str, Any] = dict(self.extra) if self.extra else {}

        # Explicit, provider-agnostic knobs win over anything placed in `extra`.
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            out["max_output_tokens"] = self.max_output_tokens

        return out