"""Fresh v11 concept-template creation and loading."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from concept_repository import SEED_PATH, atomic_write_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_template_seed() -> dict[str, Any]:
    with SEED_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "discovery" not in data:
        raise ValueError(f"Invalid concept template seed: {SEED_PATH}")
    return data


def make_blank_template(concept: str) -> dict[str, Any]:
    concept = concept.strip()
    if not concept:
        raise ValueError("Concept name is required")
    template = json.loads(json.dumps(load_template_seed()))
    template["concept"] = concept
    template["concept_id"] = f"c_{uuid.uuid4().hex[:12]}"
    template["created_at"] = now_iso()
    template["updated_at"] = now_iso()
    template.setdefault("metadata", {})["status"] = "initialized"
    for question in template.get("discovery", {}).get("questions", []):
        question["text"] = str(question.get("text", "")).replace("{concept}", concept)
        question["concept"] = concept
        question.setdefault("status", "unasked")
        question["timestamp_created"] = now_iso()
    return template


def ensure_template_file(path: str | Path, concept: str) -> dict[str, Any]:
    """Create a native v11 template or load it exactly as stored."""
    path = Path(path)
    if not path.exists():
        template = make_blank_template(concept)
        atomic_write_json(path, template)
        return template
    with path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    if not isinstance(template, dict) or template.get("concept") != concept:
        raise ValueError(f"Invalid v11 concept template: {path}")
    return template
