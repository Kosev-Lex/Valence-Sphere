"""Fresh v11 concept-directory storage and atomic JSON persistence."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from constants import PROJECT_ROOT, ROOT_DIR

GLOBAL_DIR = ROOT_DIR / "_global"
SPAWN_PATH = GLOBAL_DIR / "spawn.json"
SPAWN_LEDGER_PATH = GLOBAL_DIR / "spawn_ledger.jsonl"
SEED_PATH = PROJECT_ROOT / "concept_template_seed.json"
_LOCK_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def canonical_name(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (value or "").strip())
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        raise ValueError("Concept name cannot be empty")
    return value[:180]


def concept_slug(concept: str) -> str:
    value = safe_filename(concept).casefold().replace(" ", "_")
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
    if not value:
        raise ValueError("Concept name cannot be converted to a directory name")
    return value


def concept_dir(concept: str) -> Path:
    return ROOT_DIR / concept_slug(concept)


def concept_path(concept: str) -> Path:
    slug = concept_slug(concept)
    return ROOT_DIR / slug / f"{slug}.json"


def audit_path(concept: str, stage: str) -> Path:
    if stage not in {"socrates", "answerer", "adjudication"}:
        raise ValueError(f"Unknown audit stage: {stage}")
    slug = concept_slug(concept)
    return concept_dir(concept) / f"{stage}_{slug}.json"


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _path_lock(path: str | Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _LOCK_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def atomic_write_json(path: str | Path, data: Any) -> Path:
    """Atomically replace exactly one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with _path_lock(path):
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def update_json(path: str | Path, default: Any, updater: Callable[[Any], Any]) -> Any:
    """Read, update and atomically write a JSON document under one path lock."""
    path = Path(path)
    with _path_lock(path):
        current = load_json(path, default)
        updated = updater(json.loads(json.dumps(current)))
        atomic_write_json(path, updated)
        return updated


def save_question_constituents(path: str | Path, question_id: Any,
                               payload: dict[str, Any]) -> dict[str, Any]:
    """Write constituents into the matching question of the concept template."""
    def apply(template):
        if not isinstance(template, dict):
            raise FileNotFoundError(f"Concept JSON not found: {path}")
        questions = template.get("discovery", {}).get("questions", [])
        target = next((question for question in questions if question.get("id") == question_id), None)
        if target is None:
            raise KeyError(f"Question {question_id!r} no longer exists")
        target["constituent_analysis"] = payload
        template["updated_at"] = datetime.now(timezone.utc).isoformat()
        return template
    return update_json(path, None, apply)


def save_concept_template(path: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    """Update the concept template while retaining concurrent analysis fields."""
    path = Path(path)
    with _path_lock(path):
        prepared = json.loads(json.dumps(state))
        current = load_json(path, {})
        current_questions = {
            question.get("id"): question
            for question in current.get("discovery", {}).get("questions", [])
            if isinstance(question, dict)
        } if isinstance(current, dict) else {}
        for question in prepared.get("discovery", {}).get("questions", []):
            existing = current_questions.get(question.get("id"))
            if existing and existing.get("constituent_analysis") and not question.get("constituent_analysis"):
                question["constituent_analysis"] = existing["constituent_analysis"]
        atomic_write_json(path, prepared)
        return prepared

# https://github.com/Kosev-Lex
@dataclass(frozen=True)
class ConceptRecord:
    name: str
    directory: Path
    path: Path
    parent: str | None
    data: dict[str, Any]


def discover_concepts() -> list[ConceptRecord]:
    """Discover native v11 concept directories."""
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in sorted(ROOT_DIR.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        path = directory / f"{directory.name}.json"
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not str(data.get("concept", "")).strip():
            continue
        records.append(ConceptRecord(
            name=str(data["concept"]).strip(), directory=directory, path=path,
            parent=data.get("parent"), data=data,
        ))
    return sorted(records, key=lambda record: record.name.casefold())


def find_concept(concept: str) -> ConceptRecord | None:
    key = canonical_name(concept)
    return next((record for record in discover_concepts()
                 if canonical_name(record.name) == key), None)


def load_spawn_queue() -> list[str]:
    payload = load_json(SPAWN_PATH, [])
    if not isinstance(payload, list):
        payload = []
    values = {canonical_name(item): item.strip() for item in payload
              if isinstance(item, str) and item.strip()}
    return sorted(values.values(), key=str.casefold)


def save_spawn_queue(values: Iterable[str]) -> list[str]:
    unique = {canonical_name(value): value.strip() for value in values
              if isinstance(value, str) and value.strip()}
    queue = sorted(unique.values(), key=str.casefold)
    atomic_write_json(SPAWN_PATH, queue)
    return queue


def record_spawn_event(event: dict[str, Any]) -> None:
    GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with SPAWN_LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
