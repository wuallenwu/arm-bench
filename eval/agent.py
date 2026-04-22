"""
eval/agent.py — Pluggable agent interface for the simd-loops benchmark.

The harness owns message history, tool execution, trace recording, and
version tracking. The agent owns only one thing: given the current message
history and tool schemas, decide what to do next.

Implement BaseAgent to plug in any framework — LangChain, DSPy, a human
CLI, or Harbor's BaseAgent wrapper.

Built-in implementations:
  - LiteLLMAgent: wraps any LiteLLM-compatible model (default)

Usage:
    # Default — create implicitly from model string
    result = run_agentic_eval("loop_001", "sve", "anthropic/claude-sonnet-4-6", handle)

    # Explicit agent
    from eval.agent import LiteLLMAgent
    agent = LiteLLMAgent("anthropic/claude-sonnet-4-6", temperature=0.5)
    result = run_agentic_eval("loop_001", "sve", agent=agent, handle=handle)

    # Custom agent (e.g. a human-in-the-loop)
    class MyAgent(BaseAgent):
        def step(self, messages, tools):
            # print messages, ask a human what to do, return tool calls
            ...
    result = run_agentic_eval("loop_001", "sve", agent=MyAgent(), handle=handle)
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """A single tool call returned by the agent."""
    id: str
    name: str
    arguments: dict  # already JSON-parsed


class BaseAgent(ABC):
    """
    Minimal contract for a benchmark agent.

    The harness calls step() once per turn with the compressed message
    history (OpenAI format) and the tool schemas. The agent returns its
    reasoning text and the list of tool calls it wants to make.

    The harness then:
      1. Appends the assistant message to history (via to_assistant_message)
      2. Executes each tool call
      3. Appends tool results to history
      4. Calls step() again for the next turn

    Agents must not mutate `messages` — the harness owns the history.
    """

    @abstractmethod
    def step(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[ToolCall]]:
        """
        Decide what to do next.

        Args:
            messages: Full compressed conversation history (OpenAI format).
                      messages[0] = system, messages[1] = initial user task.
            tools:    OpenAI-compatible tool schemas (from SIMDTools.tool_schemas()).

        Returns:
            reasoning: Text the agent produced before the tool calls.
                       Should explain observation, hypothesis, and change per
                       the system prompt's reasoning requirement. May be empty.
            tool_calls: Ordered list of tool calls to execute.
                        Empty list signals the agent is done (no more actions).
        """
        ...

    def to_assistant_message(self, reasoning: str, tool_calls: list[ToolCall]) -> dict:
        """
        Convert agent output to an OpenAI-format assistant message dict
        suitable for appending to the message history.
        """
        msg: dict = {"role": "assistant", "content": reasoning or None}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
        return msg


class LiteLLMAgent(BaseAgent):
    """
    Default agent — thin wrapper around litellm.completion().

    Supports any model string that LiteLLM understands:
      anthropic/claude-sonnet-4-6
      openrouter/anthropic/claude-sonnet-4-6
      openai/gpt-4o
      ...
    """

    def __init__(self, model: str, temperature: float = 0.2, max_retries: int = 6):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def step(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[ToolCall]]:
        import litellm

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                break
            except litellm.RateLimitError as e:
                wait = 30 * (2 ** attempt)
                print(f"  [rate limit] sleeping {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError("Exceeded retry budget for rate limit")

        msg = response.choices[0].message
        reasoning = msg.content or ""

        tool_calls = []
        for tc in msg.tool_calls or []:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            ))

        return reasoning, tool_calls

    def to_assistant_message(self, reasoning: str, tool_calls: list[ToolCall]) -> dict:
        # Use the litellm response's native model_dump() format for full fidelity
        # (preserves any extra fields litellm adds, e.g. finish_reason)
        # Fall back to base class if called without a live response.
        return super().to_assistant_message(reasoning, tool_calls)
