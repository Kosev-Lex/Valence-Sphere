"""Schema-aware knowledge graph projection (independent of Tkinter)."""
from __future__ import annotations

from collections import defaultdict

from concept_repository import canonical_name


def relation_summary(template: dict) -> list[dict]:
    relations = template.get("relations") or []
    if isinstance(relations, dict):
        relations = [{"relation": key, "target": value} for key, value in relations.items()]
    return [item for item in relations if isinstance(item, dict)]


def build_concept_graph(records) -> dict:
    """Build a navigable concept-node graph from repository records."""
    records = list(records)
    by_key = {canonical_name(record.name): record for record in records}
    children = defaultdict(list)
    roots = []
    for record in records:
        parent_key = canonical_name(record.parent or "")
        if parent_key and parent_key in by_key:
            children[parent_key].append(record)
        else:
            roots.append(record)

    levels: list[list] = []
    frontier = sorted(roots, key=lambda item: item.name.casefold())
    seen = set()
    while frontier:
        level = [item for item in frontier if canonical_name(item.name) not in seen]
        if not level:
            break
        levels.append(level)
        seen.update(canonical_name(item.name) for item in level)
        frontier = []
        for item in level:
            frontier.extend(sorted(children[canonical_name(item.name)], key=lambda child: child.name.casefold()))
    remaining = [record for record in records if canonical_name(record.name) not in seen]
    if remaining:
        levels.append(sorted(remaining, key=lambda item: item.name.casefold()))

    nodes = []
    position = {}
    for level_index, level in enumerate(levels):
        for column, record in enumerate(level):
            x = 180 + column * 250
            y = 140 + level_index * 190
            key = canonical_name(record.name)
            questions = record.data.get("discovery", {}).get("questions", [])
            answered = sum(bool(question.get("answer") or question.get("answers")) for question in questions)
            node = {"id": key, "name": record.name, "path": str(record.path), "x": x, "y": y,
                    "answered": answered, "total": len(questions)}
            nodes.append(node)
            position[key] = node

    edge_keys = set()
    edges = []
    for record in records:
        source = canonical_name(record.name)
        parent = canonical_name(record.parent or "")
        if parent in position:
            edge_keys.add((parent, source, "parent"))
        for relation in relation_summary(record.data):
            target = relation.get("target") or relation.get("object") or relation.get("concept")
            target_key = canonical_name(str(target or ""))
            if target_key in position and target_key != source:
                label = str(relation.get("relation") or relation.get("type") or "related")
                edge_keys.add((source, target_key, label))
    for source, target, label in sorted(edge_keys):
        edges.append({"source": source, "target": target, "label": label})
    return {"nodes": nodes, "edges": edges}
