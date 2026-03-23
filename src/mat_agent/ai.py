from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None


DEFAULT_AGENT_MODELS = {
    "retrieval_planner": "gpt-5-mini",
    "qa_agent": "claude-sonnet-4-20250514-v1:0",
    "simulation_planner": "gpt-5",
}


@dataclass
class GatewayConfig:
    api_key: str
    base_url: str


def load_gateway_config(path: str | Path = "config.json") -> GatewayConfig | None:
    config_path = Path(path)
    if not config_path.exists():
        return None

    data = json.loads(config_path.read_text(encoding="utf-8"))
    api_key = data.get("api_key", "").strip()
    base_url = data.get("base_url", "").strip()
    if not api_key or not base_url:
        return None
    return GatewayConfig(api_key=api_key, base_url=base_url)


class AgentManager:
    def __init__(self, enable_ai: bool = True, config_path: str | Path = "config.json") -> None:
        self.enable_ai = enable_ai
        self.config_path = Path(config_path)
        self.gateway = load_gateway_config(config_path)
        self.usage_log: list[dict[str, Any]] = []
        self._client = None

    def available(self) -> bool:
        return self.enable_ai and self.gateway is not None and openai is not None

    def get_usage(self) -> list[dict[str, Any]]:
        return list(self.usage_log)

    def plan_retrieval(
        self,
        task: str,
        request_context: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        tools = [
            "local_file_reader",
            "pdf_text_extractor",
            "legacy_literature_review_handoff",
        ]
        if not self.available():
            self._record(
                "retrieval_planner",
                DEFAULT_AGENT_MODELS["retrieval_planner"],
                "skipped",
                "AI disabled or config unavailable.",
                tools,
            )
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval planning and critique agent for computational materials workflows. "
                    "Given the task, request context, and retrieved evidence, identify the best evidence focus, "
                    "coverage gaps, and critique the retrieval quality. "
                    "Return strict JSON with keys: retrieval_focus, priority_sources, coverage_assessment, "
                    "missing_information, critique."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "request_context": request_context,
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._generate_json("retrieval_planner", DEFAULT_AGENT_MODELS["retrieval_planner"], messages, tools)

    def answer_question(self, task: str, evidence: list[dict[str, Any]]) -> dict[str, Any] | None:
        tools = ["grounded_evidence_bundle", "citation_pack"]
        if not self.available():
            self._record("qa_agent", DEFAULT_AGENT_MODELS["qa_agent"], "skipped", "AI disabled or config unavailable.", tools)
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grounded materials-science QA agent. "
                    "Use only the provided evidence. Do not invent citations. "
                    "Return strict JSON with keys: answer, citations, confidence, gaps."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._generate_json("qa_agent", DEFAULT_AGENT_MODELS["qa_agent"], messages, tools)

    def plan_simulation(self, task: str, context: dict[str, Any]) -> dict[str, Any] | None:
        tools = ["retrieval_context_bundle", "software_hint_selector", "human_approval_gate"]
        if not self.available():
            self._record(
                "simulation_planner",
                DEFAULT_AGENT_MODELS["simulation_planner"],
                "skipped",
                "AI disabled or config unavailable.",
                tools,
            )
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a computational materials planning agent. "
                    "Create a practical simulation plan from the request and evidence. "
                    "Return strict JSON with keys: summary, software, requires_human_approval, parameters, assumptions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._generate_json("simulation_planner", DEFAULT_AGENT_MODELS["simulation_planner"], messages, tools)

    def _client_instance(self):
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.gateway.api_key,
                base_url=self.gateway.base_url,
            )
        return self._client

    def _generate_json(
        self,
        role: str,
        model: str,
        messages: list[dict[str, str]],
        tools: list[str],
    ) -> dict[str, Any] | None:
        try:
            client = self._client_instance()
            try:
                response = client.responses.create(model=model, messages=messages)
            except TypeError:
                response = client.responses.create(model=model, input=messages)
            text = self._extract_text(response)
            payload = self._parse_json(text)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            self._record(role, model, "fallback", self._sanitize_error(exc), tools)
            return None

        self._record(role, model, "used", "AI response applied.", tools)
        return payload

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text

        output = getattr(response, "output", None) or []
        fragments: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                if hasattr(content, "text") and content.text:
                    fragments.append(content.text)
        return "\n".join(fragments)

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise
            return json.loads(text[start : end + 1])

    def _record(self, role: str, model: str, status: str, note: str, tools: list[str]) -> None:
        self.usage_log.append(
            {
                "role": role,
                "model": model,
                "status": status,
                "note": note,
                "tools": tools,
            }
        )

    def _sanitize_error(self, exc: Exception) -> str:
        text = str(exc)
        lower = text.lower()
        if "401" in text or "authentication" in lower or "token_not_found_in_db" in lower:
            return "401 authentication error from the configured AI gateway."
        if "connection error" in lower:
            return "Connection error while calling the configured AI gateway."
        return exc.__class__.__name__
