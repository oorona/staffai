#!/usr/bin/env python3
"""Standalone evaluator for Ollama profile extraction tasks.

This script benchmarks the same JSON outputs used by the runtime user-memory
pipeline for:

- memory extraction (`{"memory": ...}`)
- style extraction (`{"style_traits": [...]}`)
- expertise extraction (`{"expertise_level": ...}`)

It does not import the application code. Instead, it loads the production
prompt templates and schemas directly from disk and talks to Ollama through the
OpenAI-compatible `/v1/chat/completions` endpoint.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request


VALID_EXPERTISE_LEVELS = {"beginner", "intermediate", "advanced"}
DEFAULT_COMPARE_MODELS = [
    "smollm2:1.7b-instruct-q4_K_M",
    "qwen3:0.6b",
    "gemma3:1b-it-qat",
    "qwen3:1.7b",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
PROMPTS_ROOT = ROOT / "utils" / "prompts"

TASK_CONFIG = {
    "memory": {
        "pack": "user_memory_tiny_extract",
    },
    "style": {
        "pack": "user_style_extract",
    },
    "expertise": {
        "pack": "user_expertise_extract",
    },
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_base_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        clean = "http://localhost:11434"
    if not clean.endswith("/v1"):
        clean = f"{clean}/v1"
    return clean


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_trait(raw: str) -> str:
    trait = collapse_spaces(raw).lower().strip(".,;:!\"'` ")
    if not trait:
        return ""
    if len(trait) > 24:
        trait = trait[:24].rstrip()
    if not re.match(r"^[a-záéíóúñü0-9][a-záéíóúñü0-9\- ]*$", trait):
        return ""
    return trait


def normalize_style_traits(raw_traits: Any) -> List[str]:
    if not isinstance(raw_traits, list):
        return []
    seen: set[str] = set()
    normalized: List[str] = []
    for raw in raw_traits:
        trait = normalize_trait(str(raw))
        if not trait or trait in seen:
            continue
        seen.add(trait)
        normalized.append(trait)
        if len(normalized) >= 5:
            break
    return normalized


def normalize_expertise(raw_level: Any) -> Optional[str]:
    level = collapse_spaces(str(raw_level or "")).lower()
    if level in VALID_EXPERTISE_LEVELS:
        return level
    return None


def normalize_memory_text(raw_memory: Any) -> str:
    text = collapse_spaces(str(raw_memory or "")).strip("`\"' ")
    text = text.strip(" .,:;")
    lowered = text.lower()
    if lowered in {"", "none", "n/a", "na"}:
        return "none"
    return text.lower()


def extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    return None
                if isinstance(parsed, dict):
                    return parsed
                return None
    return None


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    req = request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def call_ollama_for_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    response_schema: Dict[str, Any],
    timeout: float,
) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "tools": [],
        "tool_choice": "none",
        "response_format": response_schema,
    }

    try:
        raw_response = post_chat_completion(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout=timeout,
        )
        content = str(raw_response["choices"][0]["message"]["content"])
        return extract_json_object(content), content, False
    except Exception as first_error:
        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        try:
            raw_response = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                payload=fallback_payload,
                timeout=timeout,
            )
            content = str(raw_response["choices"][0]["message"]["content"])
            return extract_json_object(content), content, True
        except Exception as second_error:
            raise RuntimeError(
                f"schema_call_failed={first_error}; fallback_failed={second_error}"
            ) from second_error


def warm_up_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> float:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with valid JSON only."},
            {"role": "user", "content": 'Return exactly this JSON and nothing else: {"ready": true}'},
        ],
        "temperature": 0.0,
        "tools": [],
        "tool_choice": "none",
    }
    started_at = time.perf_counter()
    post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )
    return time.perf_counter() - started_at


def render_user_prompt(suite: str, template: str, input_text: str) -> str:
    prompt = template.replace("{{USER_MESSAGE}}", input_text)
    if suite == "style":
        prompt = prompt.replace("{{EXISTING_STYLE_TRAITS}}", "none")
    if suite == "expertise":
        prompt = prompt.replace("{{EXISTING_EXPERTISE_LEVEL}}", "none")
    return prompt


def load_task_materials() -> Dict[str, Dict[str, Any]]:
    materials: Dict[str, Dict[str, Any]] = {}
    for suite, config in TASK_CONFIG.items():
        pack_dir = PROMPTS_ROOT / str(config["pack"])
        materials[suite] = {
            "system_prompt": load_text(pack_dir / "system_prompt.txt"),
            "user_prompt_template": load_text(pack_dir / "user_prompt.txt"),
            "schema": load_json(pack_dir / "schema.json"),
        }
    return materials


def build_memory_cases() -> List[Dict[str, Any]]:
    languages = [
        "Python",
        "TypeScript",
        "Go",
        "Rust",
        "Java",
        "C#",
        "Ruby",
        "PHP",
        "Kotlin",
        "Elixir",
    ]
    frameworks = [
        "FastAPI",
        "Next.js",
        "Gin",
        "Axum",
        "Spring Boot",
        "ASP.NET Core",
        "Rails",
        "Laravel",
        "Ktor",
        "Phoenix",
    ]
    editors = [
        "Neovim",
        "VS Code",
        "IntelliJ",
        "Cursor",
        "PyCharm",
        "Helix",
        "Sublime Text",
        "Zed",
        "Emacs",
        "WebStorm",
    ]
    goals = [
        "Kubernetes",
        "Terraform",
        "GraphQL",
        "Kafka",
        "Rust",
        "PostgreSQL tuning",
        "Redis clustering",
        "gRPC",
        "CI/CD",
        "observability",
    ]
    platforms = [
        "AWS",
        "GCP",
        "Azure",
        "Fly.io",
        "Railway",
        "Render",
        "DigitalOcean",
        "Kubernetes",
        "Vercel",
        "Cloudflare",
    ]
    frontends = [
        "React",
        "Vue",
        "Svelte",
        "Angular",
        "SolidJS",
        "Alpine.js",
        "htmx",
        "Nuxt",
        "Remix",
        "Astro",
    ]
    test_tools = [
        "pytest",
        "Playwright",
        "Jest",
        "Vitest",
        "Cypress",
        "RSpec",
        "PHPUnit",
        "Go test",
        "JUnit",
        "Testcontainers",
    ]
    data_tools = [
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "SQLite",
        "ClickHouse",
        "Elasticsearch",
        "DynamoDB",
        "Snowflake",
        "BigQuery",
    ]
    protocols = [
        "Kafka",
        "RabbitMQ",
        "NATS",
        "gRPC",
        "REST",
        "WebSockets",
        "SQS",
        "Pub/Sub",
        "SNS",
        "MQTT",
    ]
    operating_systems = [
        "Ubuntu",
        "macOS",
        "Arch Linux",
        "Fedora",
        "Debian",
        "Windows",
        "Pop!_OS",
        "NixOS",
        "openSUSE",
        "Linux Mint",
    ]
    no_signal_messages = [
        "The assistant should answer in JSON and mention Redis in the docs.",
        "Our onboarding guide says the bot uses Python and Docker.",
        "The API reference says the service runs on port 8080 and stores logs in S3.",
        "The vendor documentation recommends using Terraform and AWS for the demo.",
        "The changelog says the mobile app switched from React Native to Flutter.",
        "The tutorial explains how the sample app uses PostgreSQL and Redis.",
        "The team wiki says the reference implementation is built with Go and gRPC.",
        "The runbook notes that the worker service retries failed jobs three times.",
        "The repository README says the dashboard uses Next.js and Tailwind.",
        "The documentation says the sample project deploys through GitHub Actions.",
    ]

    cases: List[Dict[str, Any]] = []
    case_id = 1

    for index, (language, framework) in enumerate(zip(languages, frameworks), start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": (
                    f"I build most of my backend work in {language} with {framework}, "
                    "and that is where I am fastest."
                ),
                "expected_output": {"memory": f"works with {language} and {framework}"},
                "validation": {"required_terms": [language.lower(), framework.lower()]},
                "notes": f"stack_pair_{index}",
            }
        )
        case_id += 1

    for index, editor in enumerate(editors, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": (
                    f"I do nearly all of my coding in {editor}; it is the editor I trust most."
                ),
                "expected_output": {"memory": f"prefers {editor}"},
                "validation": {"required_terms": [editor.lower()]},
                "notes": f"editor_preference_{index}",
            }
        )
        case_id += 1

    for index, goal in enumerate(goals, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": (
                    f"This quarter I am focused on learning {goal} so I can level up my backend work."
                ),
                "expected_output": {"memory": f"wants to learn {goal}"},
                "validation": {"required_terms": [goal.lower()]},
                "notes": f"learning_goal_{index}",
            }
        )
        case_id += 1

    for index, platform in enumerate(platforms, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": (
                    f"I deploy my projects on {platform} and that is my default hosting platform."
                ),
                "expected_output": {"memory": f"deploys on {platform}"},
                "validation": {"required_terms": [platform.lower()]},
                "notes": f"hosting_platform_{index}",
            }
        )
        case_id += 1

    for index, frontend in enumerate(frontends, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": f"Most of my frontend work lately is in {frontend}, and I use it every week.",
                "expected_output": {"memory": f"works with {frontend}"},
                "validation": {"required_terms": [frontend.lower()]},
                "notes": f"frontend_stack_{index}",
            }
        )
        case_id += 1

    for index, tool in enumerate(test_tools, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": f"My default testing tool is {tool}, and I use it on nearly every project.",
                "expected_output": {"memory": f"uses {tool}"},
                "validation": {"required_terms": [tool.lower()]},
                "notes": f"testing_tool_{index}",
            }
        )
        case_id += 1

    for index, data_tool in enumerate(data_tools, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": f"I spend a lot of time working with {data_tool} in production systems.",
                "expected_output": {"memory": f"works with {data_tool}"},
                "validation": {"required_terms": [data_tool.lower()]},
                "notes": f"data_tool_{index}",
            }
        )
        case_id += 1

    for index, protocol in enumerate(protocols, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": f"When services need to talk, I usually reach for {protocol} first.",
                "expected_output": {"memory": f"uses {protocol}"},
                "validation": {"required_terms": [protocol.lower()]},
                "notes": f"integration_pattern_{index}",
            }
        )
        case_id += 1

    for index, operating_system in enumerate(operating_systems, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": (
                    f"I do all of my development work on {operating_system}; that is my daily machine."
                ),
                "expected_output": {"memory": f"develops on {operating_system}"},
                "validation": {"required_terms": [operating_system.lower()]},
                "notes": f"os_preference_{index}",
            }
        )
        case_id += 1

    for index, message in enumerate(no_signal_messages, start=1):
        cases.append(
            {
                "suite": "memory",
                "case_id": f"memory-{case_id:03d}",
                "input_text": message,
                "expected_output": {"memory": "none"},
                "validation": {"required_terms": []},
                "notes": f"no_signal_{index}",
            }
        )
        case_id += 1

    assert len(cases) == 100
    return cases


def build_style_cases() -> List[Dict[str, Any]]:
    topics = [
        "FastAPI project structure",
        "PostgreSQL indexing",
        "Docker image size",
        "Redis cache invalidation",
        "Kubernetes rollout strategy",
        "CI pipeline failures",
        "Terraform modules",
        "Kafka consumer lag",
        "React state management",
        "logging and tracing",
    ]
    archetypes = [
        {
            "traits": ["direct", "practical", "concise"],
            "minimum_overlap": 2,
            "template": "Keep it tight on {topic}. No fluff, no preamble, just the steps that work.",
        },
        {
            "traits": ["courteous", "professional", "formal"],
            "minimum_overlap": 2,
            "template": (
                "Could you please provide a careful explanation of {topic}? "
                "I would appreciate a professional and well-structured answer."
            ),
        },
        {
            "traits": ["friendly", "casual", "playful"],
            "minimum_overlap": 2,
            "template": (
                "Hey, can you make {topic} feel less intimidating? "
                "Keep it light and easy so I can actually enjoy reading it."
            ),
        },
        {
            "traits": ["technical", "direct", "concise"],
            "minimum_overlap": 2,
            "template": (
                "For {topic}, skip the analogies and give me the technical answer in the fewest words possible."
            ),
        },
        {
            "traits": ["calm", "courteous", "professional"],
            "minimum_overlap": 2,
            "template": (
                "I am not in a rush on {topic}; please walk through it steadily and clearly without any drama."
            ),
        },
        {
            "traits": ["humorous", "playful", "casual"],
            "minimum_overlap": 2,
            "template": (
                "Teach me {topic}, but make it fun enough that I do not feel like I am reading tax law."
            ),
        },
        {
            "traits": ["sarcastic", "direct", "casual"],
            "minimum_overlap": 2,
            "template": (
                "Sure, apparently {topic} is easy, so hit me with the answer before this turns into another epic saga."
            ),
        },
        {
            "traits": ["vulgar", "direct", "casual"],
            "minimum_overlap": 2,
            "template": (
                "I am stuck on {topic}; give me the damn fix and skip the polished corporate nonsense."
            ),
        },
        {
            "traits": ["uplifting", "friendly", "courteous"],
            "minimum_overlap": 2,
            "template": (
                "I am trying to get better at {topic}; please be encouraging and help me build confidence while I learn."
            ),
        },
        {
            "traits": ["formal", "technical", "professional"],
            "minimum_overlap": 2,
            "template": (
                "Regarding {topic}, I want a precise, technically rigorous explanation with disciplined wording."
            ),
        },
    ]

    cases: List[Dict[str, Any]] = []
    case_id = 1
    for archetype in archetypes:
        for topic in topics:
            cases.append(
                {
                    "suite": "style",
                    "case_id": f"style-{case_id:03d}",
                    "input_text": str(archetype["template"]).format(topic=topic),
                    "expected_output": {"style_traits": list(archetype["traits"])},
                    "validation": {
                        "minimum_overlap": int(archetype["minimum_overlap"]),
                    },
                }
            )
            case_id += 1

    assert len(cases) == 100
    return cases


def build_expertise_cases() -> List[Dict[str, Any]]:
    beginner_topics = [
        "what a variable is",
        "what a loop does",
        "how functions work",
        "what Git is",
        "what an API means",
        "what JSON is",
        "how to run Python",
        "what a database is",
        "what HTML does",
        "what debugging means",
    ]
    intermediate_topics = [
        "database indexing strategy",
        "Docker layer caching",
        "Kubernetes deployments",
        "message queue retries",
        "CI pipeline parallelism",
        "Redis eviction policies",
        "Terraform state management",
        "load balancer health checks",
        "API rate limiting",
        "log aggregation tradeoffs",
    ]
    advanced_topics = [
        "PostgreSQL vacuum tuning",
        "eBPF latency tracing",
        "cross-region failover design",
        "distributed lock contention",
        "shard rebalancing",
        "kernel scheduler behavior",
        "consensus protocol edge cases",
        "tail latency reduction",
        "zero-downtime schema migration",
        "memory allocator fragmentation",
    ]
    templates = [
        {
            "level": "beginner",
            "topics": beginner_topics,
            "template": "I just started learning to code. Can you explain {topic} in very simple terms?",
        },
        {
            "level": "beginner",
            "topics": beginner_topics,
            "template": "This is my first week touching programming, and I still do not understand {topic}.",
        },
        {
            "level": "beginner",
            "topics": beginner_topics,
            "template": "I am completely new here, so please start from zero when talking about {topic}.",
        },
        {
            "level": "intermediate",
            "topics": intermediate_topics,
            "template": (
                "I maintain a small web service and a Postgres database, but I want a better handle on {topic}."
            ),
        },
        {
            "level": "intermediate",
            "topics": intermediate_topics,
            "template": (
                "I ship features weekly, write Dockerfiles, and debug CI, though {topic} is still an area I am improving."
            ),
        },
        {
            "level": "intermediate",
            "topics": intermediate_topics,
            "template": (
                "I am comfortable with production APIs and routine debugging, but I am still learning the details of {topic}."
            ),
        },
        {
            "level": "intermediate",
            "topics": intermediate_topics,
            "template": (
                "I can trace normal outages and fix migrations, yet I would like a deeper explanation of {topic}."
            ),
        },
        {
            "level": "advanced",
            "topics": advanced_topics,
            "template": (
                "I am tuning {topic} in production and comparing multiple mitigation strategies before the next rollout."
            ),
        },
        {
            "level": "advanced",
            "topics": advanced_topics,
            "template": (
                "I am designing around {topic} across several services and need to reason about failure domains precisely."
            ),
        },
        {
            "level": "advanced",
            "topics": advanced_topics,
            "template": (
                "I am instrumenting low-level diagnostics for {topic} and reviewing edge-case behavior under load."
            ),
        },
    ]

    cases: List[Dict[str, Any]] = []
    case_id = 1
    for template_block in templates:
        for topic in template_block["topics"]:
            cases.append(
                {
                    "suite": "expertise",
                    "case_id": f"expertise-{case_id:03d}",
                    "input_text": str(template_block["template"]).format(topic=topic),
                    "expected_output": {"expertise_level": str(template_block["level"])},
                    "validation": {},
                }
            )
            case_id += 1

    assert len(cases) == 100
    return cases


def build_all_cases() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "memory": build_memory_cases(),
        "style": build_style_cases(),
        "expertise": build_expertise_cases(),
    }


def validate_memory_case(case: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "schema_valid": False,
        "soft_pass": False,
        "exact_pass": False,
        "errors": [],
    }

    if not isinstance(parsed, dict):
        result["errors"].append("response is not a JSON object")
        return result

    memory_value = parsed.get("memory")
    if not isinstance(memory_value, str):
        result["errors"].append("missing string field: memory")
        return result

    result["schema_valid"] = True
    predicted = normalize_memory_text(memory_value)
    expected = normalize_memory_text(case["expected_output"]["memory"])

    if predicted == expected:
        result["exact_pass"] = True

    required_terms = [str(term).lower() for term in case["validation"].get("required_terms", [])]
    if expected == "none":
        result["soft_pass"] = predicted == "none"
        if not result["soft_pass"]:
            result["errors"].append("expected none")
        return result

    if predicted == "none":
        result["errors"].append("returned none for positive memory case")
        return result

    missing_terms = [term for term in required_terms if term not in predicted]
    if missing_terms:
        result["errors"].append(f"missing required memory terms: {', '.join(missing_terms)}")
        return result

    result["soft_pass"] = True
    return result


def validate_style_case(case: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "schema_valid": False,
        "soft_pass": False,
        "exact_pass": False,
        "errors": [],
    }

    if not isinstance(parsed, dict):
        result["errors"].append("response is not a JSON object")
        return result

    predicted_traits = normalize_style_traits(parsed.get("style_traits"))
    if not predicted_traits:
        result["errors"].append("invalid or empty style_traits")
        return result

    result["schema_valid"] = True
    expected_traits = normalize_style_traits(case["expected_output"]["style_traits"])
    if set(predicted_traits) == set(expected_traits):
        result["exact_pass"] = True

    overlap = len(set(predicted_traits) & set(expected_traits))
    required_overlap = int(case["validation"].get("minimum_overlap", max(1, len(expected_traits))))
    if overlap >= required_overlap:
        result["soft_pass"] = True
    else:
        result["errors"].append(
            f"style overlap too low: expected>={required_overlap}, got={overlap}"
        )
    return result


def validate_expertise_case(case: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = {
        "schema_valid": False,
        "soft_pass": False,
        "exact_pass": False,
        "errors": [],
    }

    if not isinstance(parsed, dict):
        result["errors"].append("response is not a JSON object")
        return result

    predicted = normalize_expertise(parsed.get("expertise_level"))
    if not predicted:
        result["errors"].append("invalid expertise_level")
        return result

    result["schema_valid"] = True
    expected = normalize_expertise(case["expected_output"]["expertise_level"])
    if predicted == expected:
        result["exact_pass"] = True
        result["soft_pass"] = True
    else:
        result["errors"].append(f"expected {expected}, got {predicted}")
    return result


def validate_case(case: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    suite = str(case["suite"])
    if suite == "memory":
        return validate_memory_case(case, parsed)
    if suite == "style":
        return validate_style_case(case, parsed)
    if suite == "expertise":
        return validate_expertise_case(case, parsed)
    raise ValueError(f"unknown suite: {suite}")


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def emit_progress(
    current: int,
    total: int,
    *,
    last_suite: str,
    enabled: bool,
    elapsed_seconds: float,
) -> None:
    if not enabled or total <= 0:
        return

    ratio = min(max(current / total, 0.0), 1.0)
    percent = ratio * 100.0
    width = 28
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    label = f"{last_suite}:{current}/{total}"
    avg_seconds = elapsed_seconds / current if current else 0.0
    rate = current / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_cases = max(total - current, 0)
    eta_seconds = avg_seconds * remaining_cases
    timing = (
        f"elapsed={format_duration(elapsed_seconds)}  "
        f"avg={avg_seconds:.2f}s/case  "
        f"rate={rate:.2f} case/s  "
        f"eta={format_duration(eta_seconds)}"
    )

    if sys.stderr.isatty():
        ending = "\n" if current >= total else ""
        print(
            f"\rProgress [{bar}] {percent:6.2f}%  {label}  {timing}",
            end=ending,
            flush=True,
            file=sys.stderr,
        )
        return

    should_log = current == 1 or current == total or current % 10 == 0
    if should_log:
        print(
            f"Progress [{bar}] {percent:6.2f}%  {label}  {timing}",
            file=sys.stderr,
            flush=True,
        )


def evaluate_cases(
    *,
    suite: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    max_cases: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    materials = load_task_materials()
    all_cases = build_all_cases()
    suites: Sequence[str]
    if suite == "all":
        suites = ("memory", "style", "expertise")
    else:
        suites = (suite,)

    selected_cases: List[Dict[str, Any]] = []
    for selected_suite in suites:
        selected_cases.extend(copy.deepcopy(all_cases[selected_suite]))

    if max_cases is not None:
        selected_cases = selected_cases[:max_cases]

    results: List[Dict[str, Any]] = []
    schema_fallback_count = 0
    run_started_at = time.perf_counter()

    for index, case in enumerate(selected_cases, start=1):
        suite_name = str(case["suite"])
        material = materials[suite_name]
        user_prompt = render_user_prompt(
            suite_name,
            material["user_prompt_template"],
            str(case["input_text"]),
        )
        messages = [
            {"role": "system", "content": str(material["system_prompt"])},
            {"role": "user", "content": user_prompt},
        ]
        case_started_at = time.perf_counter()

        try:
            parsed, raw_content, used_fallback = call_ollama_for_json(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                response_schema=material["schema"],
                timeout=timeout,
            )
            schema_fallback_count += int(used_fallback)
            validation = validate_case(case, parsed)
            case_result = {
                "suite": suite_name,
                "case_id": case["case_id"],
                "input_text": case["input_text"],
                "expected_output": case["expected_output"],
                "raw_content": raw_content,
                "parsed_output": parsed,
                "used_schema_fallback": used_fallback,
                **validation,
            }
        except Exception as exc:
            case_result = {
                "suite": suite_name,
                "case_id": case["case_id"],
                "input_text": case["input_text"],
                "expected_output": case["expected_output"],
                "raw_content": "",
                "parsed_output": None,
                "used_schema_fallback": False,
                "schema_valid": False,
                "soft_pass": False,
                "exact_pass": False,
                "errors": [str(exc)],
            }

        case_duration_seconds = time.perf_counter() - case_started_at
        case_result["duration_seconds"] = round(case_duration_seconds, 4)
        results.append(case_result)
        emit_progress(
            index,
            len(selected_cases),
            last_suite=suite_name,
            enabled=len(selected_cases) > 1,
            elapsed_seconds=time.perf_counter() - run_started_at,
        )
        if verbose:
            status = "PASS" if case_result["soft_pass"] else "FAIL"
            print(
                f"[{index:03d}/{len(selected_cases):03d}] {case_result['case_id']} "
                f"{suite_name} {status} {case_duration_seconds:.2f}s"
            )

    total_elapsed_seconds = time.perf_counter() - run_started_at
    average_case_seconds = total_elapsed_seconds / len(results) if results else 0.0
    throughput_cases_per_second = len(results) / total_elapsed_seconds if total_elapsed_seconds > 0 else 0.0
    summary: Dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "total_cases": len(results),
        "schema_fallback_count": schema_fallback_count,
        "timing": {
            "total_elapsed_seconds": round(total_elapsed_seconds, 4),
            "average_case_seconds": round(average_case_seconds, 4),
            "throughput_cases_per_second": round(throughput_cases_per_second, 4),
        },
        "suites": {},
        "results": results,
    }

    for suite_name in suites:
        suite_results = [item for item in results if item["suite"] == suite_name]
        total = len(suite_results)
        schema_valid = sum(1 for item in suite_results if item["schema_valid"])
        soft_pass = sum(1 for item in suite_results if item["soft_pass"])
        exact_pass = sum(1 for item in suite_results if item["exact_pass"])
        suite_total_seconds = sum(float(item.get("duration_seconds", 0.0)) for item in suite_results)
        suite_avg_seconds = suite_total_seconds / total if total else 0.0
        suite_rate = total / suite_total_seconds if suite_total_seconds > 0 else 0.0
        summary["suites"][suite_name] = {
            "total": total,
            "schema_valid": schema_valid,
            "soft_pass": soft_pass,
            "exact_pass": exact_pass,
            "total_elapsed_seconds": round(suite_total_seconds, 4),
            "average_case_seconds": round(suite_avg_seconds, 4),
            "throughput_cases_per_second": round(suite_rate, 4),
            "schema_rate": round((schema_valid / total) * 100, 2) if total else 0.0,
            "soft_rate": round((soft_pass / total) * 100, 2) if total else 0.0,
            "exact_rate": round((exact_pass / total) * 100, 2) if total else 0.0,
        }

    overall_total = len(results)
    overall_schema = sum(1 for item in results if item["schema_valid"])
    overall_soft = sum(1 for item in results if item["soft_pass"])
    overall_exact = sum(1 for item in results if item["exact_pass"])
    summary["overall"] = {
        "total": overall_total,
        "schema_valid": overall_schema,
        "soft_pass": overall_soft,
        "exact_pass": overall_exact,
        "schema_rate": round((overall_schema / overall_total) * 100, 2) if overall_total else 0.0,
        "soft_rate": round((overall_soft / overall_total) * 100, 2) if overall_total else 0.0,
        "exact_rate": round((overall_exact / overall_total) * 100, 2) if overall_total else 0.0,
    }
    return summary


def print_summary(summary: Dict[str, Any], failure_limit: int) -> None:
    timing = summary.get("timing", {})
    warmup = summary.get("warmup", {})
    print(
        f"Model: {summary['model']} | Base URL: {summary['base_url']} | "
        f"Cases: {summary['overall']['total']} | Schema fallback calls: {summary['schema_fallback_count']} | "
        f"Warmup: {format_duration(float(warmup.get('duration_seconds', 0.0)))} | "
        f"Elapsed: {format_duration(float(timing.get('total_elapsed_seconds', 0.0)))} | "
        f"Avg: {float(timing.get('average_case_seconds', 0.0)):.2f}s/case | "
        f"Rate: {float(timing.get('throughput_cases_per_second', 0.0)):.2f} case/s"
    )
    for suite_name, suite_summary in summary["suites"].items():
        print(
            f"{suite_name}: total={suite_summary['total']} "
            f"time={format_duration(float(suite_summary.get('total_elapsed_seconds', 0.0)))} "
            f"avg={float(suite_summary.get('average_case_seconds', 0.0)):.2f}s "
            f"rate={float(suite_summary.get('throughput_cases_per_second', 0.0)):.2f}/s "
            f"schema={suite_summary['schema_valid']} ({suite_summary['schema_rate']}%) "
            f"soft={suite_summary['soft_pass']} ({suite_summary['soft_rate']}%) "
            f"exact={suite_summary['exact_pass']} ({suite_summary['exact_rate']}%)"
        )
    overall = summary["overall"]
    print(
        f"overall: total={overall['total']} schema={overall['schema_valid']} ({overall['schema_rate']}%) "
        f"soft={overall['soft_pass']} ({overall['soft_rate']}%) "
        f"exact={overall['exact_pass']} ({overall['exact_rate']}%)"
    )

    failures = [item for item in summary["results"] if not item["soft_pass"]]
    if not failures:
        print("No soft validation failures.")
        return

    print("")
    print(f"Failures shown: {min(failure_limit, len(failures))}/{len(failures)}")
    for failure in failures[:failure_limit]:
        print(f"- {failure['case_id']} ({failure['suite']})")
        print(f"  input: {failure['input_text']}")
        print(f"  expected: {json.dumps(failure['expected_output'], ensure_ascii=True)}")
        if failure["parsed_output"] is not None:
            print(f"  parsed: {json.dumps(failure['parsed_output'], ensure_ascii=True)}")
        elif failure["raw_content"]:
            print(f"  raw: {failure['raw_content']}")
        if failure["errors"]:
            print(f"  errors: {' | '.join(failure['errors'])}")


def build_comparison_rows(run_summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in run_summaries:
        timing = summary.get("timing", {})
        warmup = summary.get("warmup", {})
        overall = summary.get("overall", {})
        soft_rate = float(overall.get("soft_rate", 0.0))
        exact_rate = float(overall.get("exact_rate", 0.0))
        avg_case_seconds = float(timing.get("average_case_seconds", 0.0))
        throughput = float(timing.get("throughput_cases_per_second", 0.0))
        balanced_score = ((soft_rate * 0.7) + (exact_rate * 0.3)) * throughput
        rows.append(
            {
                "model": str(summary.get("model", "")),
                "cases": int(overall.get("total", 0)),
                "soft_rate": soft_rate,
                "exact_rate": exact_rate,
                "avg_case_seconds": avg_case_seconds,
                "throughput_cases_per_second": throughput,
                "bench_seconds": float(timing.get("total_elapsed_seconds", 0.0)),
                "warmup_seconds": float(warmup.get("duration_seconds", 0.0)),
                "balanced_score": round(balanced_score, 4),
            }
        )

    accuracy_sorted = sorted(
        rows,
        key=lambda item: (
            -item["soft_rate"],
            -item["exact_rate"],
            item["avg_case_seconds"],
            item["model"],
        ),
    )
    speed_sorted = sorted(
        rows,
        key=lambda item: (
            item["avg_case_seconds"],
            -item["throughput_cases_per_second"],
            -item["soft_rate"],
            item["model"],
        ),
    )
    balanced_sorted = sorted(
        rows,
        key=lambda item: (
            -item["balanced_score"],
            -item["soft_rate"],
            item["avg_case_seconds"],
            item["model"],
        ),
    )

    accuracy_rank = {row["model"]: index for index, row in enumerate(accuracy_sorted, start=1)}
    speed_rank = {row["model"]: index for index, row in enumerate(speed_sorted, start=1)}
    balanced_rank = {row["model"]: index for index, row in enumerate(balanced_sorted, start=1)}

    for row in rows:
        row["accuracy_rank"] = accuracy_rank[row["model"]]
        row["speed_rank"] = speed_rank[row["model"]]
        row["balanced_rank"] = balanced_rank[row["model"]]

    rows.sort(key=lambda item: (item["balanced_rank"], item["accuracy_rank"], item["speed_rank"], item["model"]))
    return rows


def print_comparison_table(comparison_rows: Sequence[Dict[str, Any]]) -> None:
    if not comparison_rows:
        return

    rows: List[List[str]] = []
    for row in comparison_rows:
        rows.append(
            [
                str(row["balanced_rank"]),
                str(row["accuracy_rank"]),
                str(row["speed_rank"]),
                str(row["model"]),
                str(row["cases"]),
                f"{float(row['soft_rate']):.2f}%",
                f"{float(row['exact_rate']):.2f}%",
                f"{float(row['avg_case_seconds']):.2f}s",
                f"{float(row['throughput_cases_per_second']):.2f}/s",
                format_duration(float(row["bench_seconds"])),
                format_duration(float(row["warmup_seconds"])),
                f"{float(row['balanced_score']):.2f}",
            ]
        )

    headers = ["Rk", "Acc", "Spd", "Model", "Cases", "Soft%", "Exact%", "Avg", "Rate", "Bench", "Warmup", "Blend"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)

    print("")
    print("Comparison")
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


def write_comparison_csv(path: Path, comparison_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "balanced_rank",
        "accuracy_rank",
        "speed_rank",
        "model",
        "cases",
        "soft_rate",
        "exact_rate",
        "avg_case_seconds",
        "throughput_cases_per_second",
        "bench_seconds",
        "warmup_seconds",
        "balanced_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_cases(path: Path, cases_by_suite: Dict[str, List[Dict[str, Any]]]) -> None:
    payload = {
        "generated_by": "standalone/ollama_profile_eval.py",
        "suite_counts": {suite: len(cases) for suite, cases in cases_by_suite.items()},
        "cases": cases_by_suite,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama against the repo's memory/style/expertise JSON contracts."
    )
    parser.add_argument(
        "--suite",
        choices=("memory", "style", "expertise", "all"),
        default="all",
        help="Which suite to run. 'all' runs 300 total cases.",
    )
    parser.add_argument(
        "--model",
        help="Single Ollama model name to benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Run the benchmark sequentially for multiple models.",
    )
    parser.add_argument(
        "--compare-defaults",
        action="store_true",
        help=(
            "Run the benchmark for the built-in comparison set: "
            + ", ".join(DEFAULT_COMPARE_MODELS)
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("USER_MEMORY_OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
        help="Ollama base URL. '/v1' is appended automatically when needed.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("USER_MEMORY_OLLAMA_API_KEY", "ollama"),
        help="API key for the OpenAI-compatible Ollama endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("USER_MEMORY_OLLAMA_TIMEOUT_S", "30")),
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional global case cap after suite selection (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--write-cases",
        type=Path,
        help="Write the generated test cases (text + expected output) to a JSON file and exit.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        help="Write the full evaluation report JSON to a file.",
    )
    parser.add_argument(
        "--write-comparison-csv",
        type=Path,
        help="Write the model comparison summary to CSV.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many failed cases to print after the summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each case status as it runs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    cases_by_suite = build_all_cases()
    selected_models: List[str] = []

    if args.compare_defaults:
        selected_models.extend(DEFAULT_COMPARE_MODELS)
    if args.models:
        selected_models.extend(args.models)
    if args.model:
        selected_models.append(args.model)
    selected_models = list(dict.fromkeys(selected_models))

    if args.write_cases:
        write_cases(args.write_cases, cases_by_suite)
        print(f"Wrote generated cases to {args.write_cases}")
        if not selected_models:
            return 0

    if not selected_models:
        print(
            "--model, --models, or --compare-defaults is required unless you only use --write-cases",
            file=sys.stderr,
        )
        return 2

    base_url = normalize_base_url(args.base_url)
    run_summaries: List[Dict[str, Any]] = []

    for model_name in selected_models:
        try:
            print(f"Warming up model: {model_name}", file=sys.stderr, flush=True)
            warmup_seconds = warm_up_model(
                base_url=base_url,
                api_key=args.api_key,
                model=model_name,
                timeout=args.timeout,
            )
        except error.URLError as exc:
            print(f"Network error talking to Ollama during warmup: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Warmup failed for {model_name}: {exc}", file=sys.stderr)
            return 1

        try:
            summary = evaluate_cases(
                suite=args.suite,
                model=model_name,
                base_url=base_url,
                api_key=args.api_key,
                timeout=args.timeout,
                max_cases=args.max_cases,
                verbose=bool(args.verbose),
            )
            summary["warmup"] = {
                "duration_seconds": round(warmup_seconds, 4),
            }
        except error.URLError as exc:
            print(f"Network error talking to Ollama: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Evaluation failed: {exc}", file=sys.stderr)
            return 1

        run_summaries.append(summary)
        print("")
        print_summary(summary, args.show_failures)

    comparison_rows = build_comparison_rows(run_summaries)
    if len(run_summaries) > 1:
        print_comparison_table(comparison_rows)

    report_payload: Dict[str, Any]
    if len(run_summaries) == 1:
        report_payload = run_summaries[0]
    else:
        report_payload = {
            "base_url": base_url,
            "suite": args.suite,
            "models": selected_models,
            "comparison": comparison_rows,
            "runs": run_summaries,
        }

    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(
            json.dumps(report_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print("")
        print(f"Wrote report to {args.write_report}")

    if args.write_comparison_csv:
        write_comparison_csv(args.write_comparison_csv, comparison_rows)
        print("")
        print(f"Wrote comparison CSV to {args.write_comparison_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
