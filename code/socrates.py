"""Socrates: ordered concept-template Questioner."""
# aka questioner module

from __future__ import annotations

from typing import Any


class Questioner:
    """Selects the next focused concept question without generating duplicates."""
    def __init__(self, concept: str, coverage_ref: dict[str, float] | None = None):
        self.concept = concept
        self.coverage = dict(coverage_ref or {})
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def definition_context(template: dict) -> str:
        for question in template.get("discovery", {}).get("questions", []):
            if (str(question.get("text", "")).strip().casefold().startswith("what is ")
                    and question.get("status") == "answered" and question.get("answer")):
                return str(question["answer"]).strip()
        return ""

    def next_question(self, template: dict) -> dict | None:
        questions = template.get("discovery", {}).get("questions", [])
        pending = [question for question in questions if question.get("status", "unasked") != "answered"]
        if not pending:
            return None
        # The focused 20-question order is intentional.
        question = pending[0]
        result = dict(question)
        result["concept"] = self.concept
        context = self.definition_context(template)
        if context and not str(result.get("text", "")).strip().casefold().startswith("what is "):
            result["context"] = {"type": "accepted_definition", "text": context}
        self.history.append(result)
        return result

    # https://github.com/Kosev-Lex
    def ask(self, last_answer=None, template: dict | None = None) -> dict:
        """Compatibility entry point; callers should supply the concept template."""
        if template is None:
            from concept_repository import find_concept
            record = find_concept(self.concept)
            template = record.data if record else {"discovery": {"questions": []}}
        question = self.next_question(template)
        if question is None:
            return {"concept": self.concept, "text": "", "status": "complete"}
        return question

    def mark_phase_learned(self, phase: str, boost: float = 0.25):
        self.coverage[phase] = min(1.0, self.coverage.get(phase, 0.0) + boost)
