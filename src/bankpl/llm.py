"""Model clients: Anthropic tool use, and a scripted model for tests and offline demos."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-haiku-5-5"


def load_dotenv(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip("'\"")
        if value and not os.environ.get(key):
            os.environ[key] = value


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: list = field(default_factory=list)  # provider-format assistant content


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        load_dotenv()
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set — add it to .env (see .env.example).")
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL

    def chat(self, system: str, messages: list, tools: list) -> Reply:
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=2048, system=system, messages=messages, tools=tools
            )
        except Exception as e:
            if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                raise SystemExit(
                    "API key rejected — create a new key in the console and put it in .env."
                ) from None
            raise
        content = [b.model_dump(exclude_none=True) for b in msg.content]
        text = "".join(b.get("text", "") for b in content if b["type"] == "text")
        calls = [
            ToolCall(b["id"], b["name"], b.get("input") or {})
            for b in content
            if b["type"] == "tool_use"
        ]
        return Reply(text, calls, content)


class ScriptedLLM:
    """Replays a fixed list of steps. Each step is either a string (final text) or a list
    of (tool_name, args) tuples, or a function of the messages returning either. Used by
    the tests to drive the agent deterministically."""

    def __init__(self, steps: list):
        self.steps = list(steps)
        self.seen: list[list] = []

    def chat(self, system: str, messages: list, tools: list) -> Reply:
        self.seen.append(messages[-1]["content"])
        step = self.steps.pop(0) if self.steps else "OK."
        if callable(step):  # a step may look at the conversation, e.g. to reuse an action id
            step = step(messages)
        if isinstance(step, str):
            return Reply(step, [], [{"type": "text", "text": step}])
        calls = [ToolCall(f"t{len(self.seen)}_{i}", n, a) for i, (n, a) in enumerate(step)]
        content = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in calls]
        return Reply("", calls, content)
