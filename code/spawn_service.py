"""Canonical spawn queue and concept creation service."""
from __future__ import annotations

from concept_repository import (
    atomic_write_json, concept_path, find_concept, load_spawn_queue,
    record_spawn_event, save_spawn_queue,
)
from template_schema import make_blank_template
from utils import normalize_concept_name


class SpawnService:
    def candidates(self) -> list[str]:
        return load_spawn_queue()

    def remove(self, values: list[str]) -> list[str]:
        rejected = {value.casefold() for value in values}
        return save_spawn_queue([item for item in self.candidates() if item.casefold() not in rejected])

    def spawn_all(self) -> list[str]:
        queue = self.candidates()
        completed = []
        created = []
        for seed in queue:
            concept = normalize_concept_name(seed)
            if not concept:
                continue
            existing = find_concept(concept)
            if existing:
                completed.append(seed)
                record_spawn_event({"seed": seed, "concept": concept, "status": "already_exists",
                                    "path": str(existing.path)})
                continue
            path = concept_path(concept)
            template = make_blank_template(concept)
            template.setdefault("metadata", {})["origin_source"] = "spawn.json"
            atomic_write_json(path, template)
            completed.append(seed)
            created.append(concept)
            record_spawn_event({"seed": seed, "concept": concept, "status": "spawned", "path": str(path)})
        save_spawn_queue([seed for seed in queue if seed not in completed])
        return created
