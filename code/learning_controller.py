"""Deterministic Questioner → Answerer → Adjudicator learning pipeline."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from adjudicator import Adjudicator
from concept_repository import find_concept, save_concept_template
from plato import Answerer, InformationSource
from socrates import Questioner
from template_schema import ensure_template_file, now_iso


class ConceptMemoryView:
    """Read-only knowledge view used by the Adjudicator."""
    def get_fact_text(self, concept: str, role: str):
        if role != "what":
            return None
        record = find_concept(concept)
        if not record:
            return None
        for question in record.data.get("discovery", {}).get("questions", []):
            if str(question.get("text", "")).casefold().startswith("what is ") and question.get("answer"):
                return question["answer"]
        return None

    def ensure_concept(self, _concept: str):
        return None


@dataclass
class LearningResult:
    decision: str
    concept: str
    question: dict[str, Any]
    answer: dict[str, Any]
    verdict: dict[str, Any]
    coverage: dict[str, float]
    error: str | None = None


class LearningController:
    """Coordinates learning without GUI, graph, spawn, or reasoning concerns."""
    def __init__(self, concept: str, model: str | None = None, temperature: float = 0.2,
                 event_sink: Callable[[str, dict], None] | None = None):
        self.concept = concept.strip()
        if not self.concept:
            raise ValueError("Concept is required")
        record = find_concept(self.concept)
        if record:
            self.path = record.path
        else:
            from concept_repository import concept_path
            self.path = concept_path(self.concept)
        self.state = ensure_template_file(str(self.path), self.concept)
        self.questioner = Questioner(self.concept)
        self.answerer = Answerer(InformationSource(model=model, temperature=temperature))
        self.adjudicator = Adjudicator(ConceptMemoryView())
        self.event_sink = event_sink
        self.stop_event = threading.Event()

    def emit(self, event: str, **payload):
        if self.event_sink:
            self.event_sink(event, payload)

    def reload(self) -> dict:
        self.state = ensure_template_file(str(self.path), self.concept)
        return self.state

    def definition_context(self) -> str:
        return self.questioner.definition_context(self.state)

    @staticmethod
    def coverage(template: dict) -> dict[str, float]:
        phases: dict[str, list[bool]] = {}
        for question in template.get("discovery", {}).get("questions", []):
            phases.setdefault(question.get("phase", "discovery"), []).append(question.get("status") == "answered")
        return {phase: sum(values) / len(values) for phase, values in phases.items()}

    def next_question(self) -> dict | None:
        self.reload()
        return self.questioner.next_question(self.state)

    def answer_question(self, question: dict) -> LearningResult:
        self.reload()
        stored = self._find_question(question)
        if stored is not None:
            stored["status"] = "asked"
            stored["timestamp_asked"] = now_iso()
            self._save()

        qobj = dict(question)
        qobj.setdefault("concept", self.concept)
        qobj.setdefault("prep", {"endpoint": "openai.chat.completions", "params": {}})
        definition = self.definition_context()
        if definition and not str(qobj.get("text", "")).casefold().startswith("what is "):
            qobj["context"] = {"type": "accepted_definition", "text": definition}
        self.emit("question", question=qobj)

        answer = self.answerer.answer(qobj)
        if answer.get("error"):
            if stored is not None:
                stored["status"] = "unasked"
                self._save()
            result = LearningResult("error", self.concept, qobj, answer, {}, self.coverage(self.state),
                                    error=answer["error"])
            self.emit("error", error=result.error)
            return result

        verdict = self.adjudicator.judge(qobj, answer)
        decision = verdict.get("decision", "pending")
        self.reload()
        stored = self._find_question(qobj)
        if stored is None:
            stored = dict(qobj)
            self.state.setdefault("discovery", {}).setdefault("questions", []).append(stored)

        audit = {
            "timestamp": now_iso(), "question_id": qobj.get("id"), "question": qobj.get("text"),
            "answer": answer.get("answer_text", ""), "verdict": verdict,
            "context_used": answer.get("context_used", ""),
        }
        self.state.setdefault("adjudication_history", []).append(audit)
        if decision == "accepted":
            timestamp = now_iso()
            stored["answer"] = answer.get("answer_text", "")
            stored.setdefault("answers", []).append({
                "text": answer.get("answer_text", ""), "decision": "accepted",
                "confidence": verdict.get("confidence"), "timestamp": timestamp,
                "context_used": answer.get("context_used", ""),
            })
            stored["status"] = "answered"
            stored["timestamp_answered"] = timestamp
            stored["confidence"] = verdict.get("confidence")
            stored["source"] = answer.get("provider_payload", {}).get("provider", "")
        else:
            stored["status"] = "unasked" if decision in {"pending", "rejected"} else "review"
            self.state.setdefault("review_queue", []).append(audit)
        self.state["coverage"] = self.coverage(self.state)
        self._save()
        result = LearningResult(decision, self.concept, qobj, answer, verdict,
                                self.state.get("coverage", {}))
        self.emit("result", result=result)
        return result

    def learn_all(self, max_questions: int | None = None) -> list[LearningResult]:
        self.stop_event.clear()
        results = []
        limit = max_questions or len(self.state.get("discovery", {}).get("questions", []))
        while len(results) < limit and not self.stop_event.is_set():
            question = self.next_question()
            if not question:
                break
            result = self.answer_question(question)
            results.append(result)
            # A pending/rejected answer leaves the question active for human
            # review; retrying it immediately would repeat the same API call.
            if result.error or result.decision != "accepted":
                break
        return results

    def stop(self):
        self.stop_event.set()

    def _find_question(self, question: dict):
        questions = self.state.setdefault("discovery", {}).setdefault("questions", [])
        qid = question.get("id")
        text = question.get("text")
        return next((item for item in questions if (qid is not None and item.get("id") == qid)
                     or (text and item.get("text") == text)), None)

    def _save(self):
        self.state["updated_at"] = now_iso()
        self.state = save_concept_template(self.path, self.state)
