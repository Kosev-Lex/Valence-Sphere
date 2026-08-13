"""Background ValenceSphere fact-checking pipeline for ordinary Stage 2 chat."""
from __future__ import annotations

import json
import re
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from concept_repository import audit_path, canonical_name, discover_concepts, load_json, update_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_assertion(text: str) -> str:
    return " ".join(re.findall(r"[\w']+", (text or "").casefold()))


def tokens(text: str) -> set[str]:
    stop = {"a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "and",
            "or", "in", "on", "for", "with", "that", "this", "it", "its", "do", "does"}
    return {word for word in re.findall(r"[\w']+", (text or "").casefold())
            if len(word) > 1 and word not in stop}


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError("The model did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object")
    return value


def parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.S)
        if not match:
            raise ValueError("The scanner did not return a JSON array")
        value = json.loads(match.group(0))
    if not isinstance(value, list):
        raise ValueError("The scanner response must be a JSON array")
    return [item for item in value if isinstance(item, dict)]


CONFIDENCE_CATEGORIES = (
    (85.0, "correct", "#22c55e", "Verified correct"),
    (65.0, "likely_correct", "#86efac", "Likely correct"),
    (40.0, "uncertain", "#facc15", "Mixed or uncertain"),
    (20.0, "likely_incorrect", "#fb923c", "Likely incorrect"),
    (0.0, "incorrect", "#ef4444", "Incorrect"),
)


def confidence_category(score: float) -> dict[str, Any]:
    score = max(0.0, min(100.0, float(score)))
    for floor, key, colour, label in CONFIDENCE_CATEGORIES:
        if score >= floor:
            return {"key": key, "label": label, "colour": colour,
                    "minimum": floor, "score": round(score, 1)}
    raise AssertionError("confidence category table is incomplete")


class WebSearchAdapter:
    """Extension point for a future application-owned web-search tool."""

    available = False

    def search(self, assertion: str, question: str, limit: int = 3) -> list[dict[str, str]]:
        return []


@dataclass
class FactCheckOutcome:
    concept: str
    assertion: str
    subject: str
    question: str
    cached: bool
    answerer: dict[str, Any]
    adjudication: dict[str, Any] | None


class FactCheckEngine:
    """Scans paragraphs and creates cumulative concept-local audit records."""

    SCANNER_SYSTEM = """You scan one chat paragraph for factual assertions.
Return a JSON array only. Each factual assertion must be an object with exactly:
subject, assertion, question, concept_hint, asserted_value.
subject is a highly compressed retrieval phrase, normally 2-5 words, such as "lemon colour".
assertion preserves the checkable claim, such as "Lemons are red".
question converts it into a direct verification question, such as "Are lemons red?".
concept_hint is the principal entity, such as "lemon". asserted_value is the claimed value, such as "red".
Do not include requests, opinions, greetings, fiction presented as fiction, or non-checkable statements.
If there are no factual assertions return []."""

    ANSWERER_SYSTEM = """You are the ValenceSphere Answerer fact-comparison module.
Compare the assertion and verification question only with the evidence supplied by Socrates.
Return JSON only with: relationship, direct_answer, assertion_support, reasoning, corrected_fact, flag.
relationship must be supports, contradicts, or insufficient.
assertion_support is 0-100 support for the assertion from the supplied evidence.
flag is correct, incorrect, or verify. Do not use outside knowledge."""

    SOURCE_SYSTEM = """You independently verify one factual assertion.
Return JSON only with: position_score, confidence, verdict, explanation, corrected_fact, sources.
position_score is 0 when the assertion is contradicted, 50 when uncertain, and 100 when supported.
confidence is your confidence in that position from 0-100.
verdict is supports, contradicts, or uncertain. sources is an array of objects with title and url.
If you have native web search, use it and record its sources. If not, say so in explanation and use your
best factual assessment without inventing citations. Never fabricate a URL."""

    ADJUDICATOR_SYSTEM = """You are the final reporting layer of a deterministic fact-check.
The numerical spectrum and category are already calculated and must not be changed.
Return JSON only with: summary, corrected_fact, note.
Summarize why the sources support or contradict the assertion. Prefer the supplied corrected fact.
Do not add facts or citations absent from the source results."""

    def __init__(self, client_factory: Callable[[str], Any],
                 web_search: WebSearchAdapter | None = None,
                 event_sink: Callable[[str, dict[str, Any]], None] | None = None):
        self.client_factory = client_factory
        self.web_search = web_search or WebSearchAdapter()
        self.event_sink = event_sink

    def emit(self, event: str, **payload):
        if self.event_sink:
            self.event_sink(event, payload)

    def scan_message(self, text: str, speaker: str, message_id: str,
                     contexts: dict[str, dict[str, Any]] | None = None) -> list[FactCheckOutcome]:
        contexts = contexts or {}
        outcomes = []
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs, 1):
            assertions = self._extract_assertions(paragraph, contexts)
            for assertion_index, claim in enumerate(assertions, 1):
                outcome = self._process_assertion(
                    claim, paragraph, speaker, message_id, paragraph_index, assertion_index, contexts)
                if outcome:
                    outcomes.append(outcome)
                    self.emit("outcome", outcome=outcome)
        return outcomes

    def _complete(self, role: str, system: str, user: str,
                  contexts: dict[str, dict[str, Any]]):
        context = contexts.get(role, {})
        guidance = str(context.get("guidance", "")).strip()
        if guidance:
            system = f"{system}\n\nUSER-SUPPLIED {role.upper()} GUIDANCE:\n{guidance}"
        attachments = context.get("attachments", [])
        return self.client_factory(role).complete(system, user, attachments)

    def _extract_assertions(self, paragraph: str,
                            contexts: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
        response = self._complete("questioner", self.SCANNER_SYSTEM, paragraph, contexts)
        assertions = []
        for item in parse_json_array(response.text):
            prepared = {key: str(item.get(key, "")).strip() for key in
                        ("subject", "assertion", "question", "concept_hint", "asserted_value")}
            if prepared["assertion"] and prepared["question"] and prepared["subject"]:
                assertions.append(prepared)
        return assertions

    def _process_assertion(self, claim: dict[str, str], paragraph: str, speaker: str,
                           message_id: str, paragraph_index: int,
                           assertion_index: int,
                           contexts: dict[str, dict[str, Any]]) -> FactCheckOutcome | None:
        record = self._closest_concept(claim)
        if not record:
            self.emit("unmatched", claim=claim, message_id=message_id)
            return None
        concept = record.name
        assertion_id = f"fc_{uuid.uuid4().hex[:12]}"
        prior = self._cached_adjudication(concept, claim["assertion"])
        evidence = self._retrieve_evidence(record.data, claim)
        socrates = {
            "assertion_id": assertion_id,
            "timestamp": now_iso(),
            "message": {"id": message_id, "speaker": speaker,
                        "paragraph": paragraph_index, "assertion": assertion_index,
                        "text": paragraph},
            "concept": concept,
            "subject": claim["subject"],
            "assertion": claim["assertion"],
            "assertion_key": normalize_assertion(claim["assertion"]),
            "question": claim["question"],
            "asserted_value": claim["asserted_value"],
            "evidence": evidence,
            "cache_hit": prior is not None,
            "prior_adjudication": prior,
        }
        self._append_ledger(concept, "socrates", socrates)
        self.emit("socrates", record=socrates)

        if prior:
            answer_result = self._answer_from_cache(prior)
        else:
            answer_result = self._answer_from_evidence(socrates, contexts)
        answerer = {
            "assertion_id": assertion_id, "timestamp": now_iso(),
            "questioner": socrates, "answerer": answer_result,
            "verification_required": not bool(prior),
        }
        self._append_ledger(concept, "answerer", answerer)
        self.emit("answerer", record=answerer)

        adjudication = prior
        if not prior:
            adjudication = self._adjudicate(concept, answerer, contexts)
        return FactCheckOutcome(
            concept=concept, assertion=claim["assertion"], subject=claim["subject"],
            question=claim["question"], cached=bool(prior), answerer=answerer,
            adjudication=adjudication,
        )

    @staticmethod
    def _closest_concept(claim: dict[str, str]):
        query = tokens(" ".join((claim["subject"], claim["assertion"], claim["concept_hint"])))
        best = None
        best_score = 0.0
        hint = canonical_name(claim["concept_hint"])
        for record in discover_concepts():
            name_tokens = tokens(record.name)
            corpus_tokens = tokens(json.dumps(record.data, ensure_ascii=False))
            exact = 5.0 if hint and hint == canonical_name(record.name) else 0.0
            name_overlap = len(query & name_tokens) / max(1, len(name_tokens)) * 3.0
            corpus_overlap = len(query & corpus_tokens) / max(1, len(query))
            score = exact + name_overlap + corpus_overlap
            if score > best_score:
                best, best_score = record, score
        return best if best_score >= 0.35 else None

    @staticmethod
    def _retrieve_evidence(template: dict[str, Any], claim: dict[str, str]) -> list[dict[str, Any]]:
        query = tokens(" ".join((claim["subject"], claim["assertion"], claim["question"])))
        grouped: dict[str, dict[str, Any]] = {}
        for question in template.get("discovery", {}).get("questions", []):
            if question.get("status") != "answered" or not question.get("answer"):
                continue
            answer = str(question["answer"]).strip()
            constituent = question.get("constituent_analysis") or {}
            searchable = " ".join((str(question.get("text", "")), answer,
                                   json.dumps(constituent, ensure_ascii=False)))
            overlap = len(query & tokens(searchable)) / max(1, len(query))
            if overlap <= 0:
                continue
            key = normalize_assertion(answer)
            confidence = float(question.get("confidence") or 0.0)
            if key in grouped:
                grouped[key]["occurrences"] += 1
                grouped[key]["confidence"] = max(grouped[key]["confidence"], confidence)
                grouped[key]["relevance"] = max(grouped[key]["relevance"], overlap)
                grouped[key]["question_ids"].append(question.get("id"))
            else:
                grouped[key] = {
                    "context_used": answer,
                    "occurrences": 1,
                    "confidence": confidence,
                    "relevance": round(overlap * 100, 1),
                    "section": question.get("phase", "discovery"),
                    "question_ids": [question.get("id")],
                }
        return sorted(grouped.values(), key=lambda item: item["relevance"], reverse=True)[:6]

    @staticmethod
    def _ledger_default(concept: str, stage: str) -> dict[str, Any]:
        return {"schema_version": 1, "concept": concept, "stage": stage,
                "updated_at": now_iso(), "records": []}

    def _append_ledger(self, concept: str, stage: str, record: dict[str, Any]):
        path = audit_path(concept, stage)
        def append(ledger):
            if not isinstance(ledger, dict):
                ledger = self._ledger_default(concept, stage)
            ledger.setdefault("records", []).append(record)
            ledger["updated_at"] = now_iso()
            return ledger
        update_json(path, self._ledger_default(concept, stage), append)

    @staticmethod
    def _cached_adjudication(concept: str, assertion: str) -> dict[str, Any] | None:
        path = audit_path(concept, "adjudication")
        ledger = load_json(path, {})
        key = normalize_assertion(assertion)
        records = ledger.get("records", []) if isinstance(ledger, dict) else []
        for record in reversed(records):
            if record.get("assertion_key") == key and record.get("final"):
                return record
        return None

    @staticmethod
    def _answer_from_cache(prior: dict[str, Any]) -> dict[str, Any]:
        final = prior.get("final", {})
        decision = final.get("category", {}).get("key", "uncertain")
        support = float(final.get("assertion_support", 50.0))
        return {
            "mode": "cached_adjudication",
            "relationship": "supports" if support >= 65 else "contradicts" if support < 40 else "insufficient",
            "direct_answer": final.get("summary", "Previously adjudicated assertion."),
            "assertion_support": support,
            "reasoning": "A matching completed adjudication was reused; external verification was skipped.",
            "corrected_fact": final.get("corrected_fact", ""),
            "flag": decision,
            "reused_assertion_id": prior.get("assertion_id"),
        }

    def _answer_from_evidence(self, socrates: dict[str, Any],
                              contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not socrates["evidence"]:
            return {"mode": "template_comparison", "relationship": "insufficient",
                    "direct_answer": "The concept template contains no relevant accepted evidence.",
                    "assertion_support": 50.0, "reasoning": "No relevant evidence was retrieved.",
                    "corrected_fact": "", "flag": "verify"}
        response = self._complete("answerer", self.ANSWERER_SYSTEM,
                                  json.dumps(socrates, indent=2, ensure_ascii=False), contexts)
        result = parse_json_object(response.text)
        return {
            "mode": "template_comparison",
            "relationship": str(result.get("relationship", "insufficient")),
            "direct_answer": str(result.get("direct_answer", "")),
            "assertion_support": max(0.0, min(100.0, float(result.get("assertion_support", 50.0)))),
            "reasoning": str(result.get("reasoning", "")),
            "corrected_fact": str(result.get("corrected_fact", "")),
            "flag": str(result.get("flag", "verify")),
        }

    def _source_check(self, role: str, assertion: str, question: str,
                      contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        web_results = self.web_search.search(assertion, question, limit=3) if self.web_search.available else []
        prompt = json.dumps({"assertion": assertion, "question": question,
                             "web_search_adapter_results": web_results}, ensure_ascii=False)
        # Source models share the Adjudicator's added evidence/attachments.
        source_contexts = dict(contexts)
        source_contexts[role] = contexts.get("adjudicator", {})
        client = self.client_factory(role)
        system = self.SOURCE_SYSTEM
        guidance = str(source_contexts[role].get("guidance", "")).strip()
        if guidance:
            system += f"\n\nUSER-SUPPLIED {role.upper()} GUIDANCE:\n{guidance}"
        response = client.complete(system, prompt, source_contexts[role].get("attachments", []))
        result = parse_json_object(response.text)
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        valid_sources = []
        for source in sources:
            if isinstance(source, dict):
                title, url = str(source.get("title", "")).strip(), str(source.get("url", "")).strip()
                if title or url:
                    valid_sources.append({"title": title, "url": url})
        return {
            "source_role": role,
            "model": response.model,
            "position_score": max(0.0, min(100.0, float(result.get("position_score", 50.0)))),
            "confidence": max(0.0, min(100.0, float(result.get("confidence", 50.0)))),
            "verdict": str(result.get("verdict", "uncertain")),
            "explanation": str(result.get("explanation", "")),
            "corrected_fact": str(result.get("corrected_fact", "")),
            "sources": valid_sources,
            "search_mode": ("web_adapter" if web_results else
                            "native_web_search_requested" if getattr(client, "profile", {}).get("native_web_search")
                            else "independent_model_assessment"),
            "usage": response.usage,
        }

    def _adjudicate(self, concept: str, answerer: dict[str, Any],
                    contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        questioner = answerer["questioner"]
        assertion, question = questioner["assertion"], questioner["question"]
        results = []
        errors = []
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="vs-source") as executor:
            futures = {executor.submit(self._source_check, role, assertion, question, contexts): role
                       for role in ("source_1", "source_2", "source_3")}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"source_role": futures[future], "error": str(exc)})
        if len(results) != 3:
            detail = "; ".join(item["error"] for item in errors) or "unknown source failure"
            raise RuntimeError(f"Verification requires all 3 sources; {len(results)} completed: {detail}")
        results.sort(key=lambda item: item["source_role"])
        models = [str(item.get("model", "")).strip().casefold() for item in results]
        if len(set(models)) != 3:
            raise RuntimeError("Verification requires 3 different source models; configure Source 1, 2 and 3 separately.")
        scores = [item["position_score"] for item in results]
        average = statistics.fmean(scores)
        spread = statistics.pstdev(scores) if len(scores) > 1 else 50.0
        source_confidence = statistics.fmean(item["confidence"] for item in results)
        agreement = max(0.0, 100.0 - spread * 2.0)
        verification_confidence = (source_confidence + agreement) / 2.0
        category = confidence_category(average)
        corrected_candidates = [item["corrected_fact"].strip() for item in results
                                if item["corrected_fact"].strip()]
        corrected_fact = max(corrected_candidates, key=corrected_candidates.count) \
            if corrected_candidates else answerer["answerer"].get("corrected_fact", "")
        deterministic = {
            "assertion_support": round(average, 1),
            "verification_confidence": round(verification_confidence, 1),
            "source_agreement": round(agreement, 1),
            "category": category,
            "corrected_fact": corrected_fact,
        }
        try:
            synthesis_response = self._complete(
                "adjudicator", self.ADJUDICATOR_SYSTEM,
                json.dumps({"assertion": assertion, "calculated": deterministic,
                            "source_results": results}, indent=2, ensure_ascii=False), contexts)
            synthesis = parse_json_object(synthesis_response.text)
        except Exception as exc:
            synthesis = {"summary": category["label"], "corrected_fact": corrected_fact,
                         "note": f"Deterministic result used; synthesis unavailable: {exc}"}
        final = {
            **deterministic,
            "summary": str(synthesis.get("summary", category["label"])),
            "corrected_fact": str(synthesis.get("corrected_fact") or corrected_fact),
            "note": str(synthesis.get("note", "")),
        }
        adjudication = {
            "assertion_id": questioner["assertion_id"], "timestamp": now_iso(),
            "concept": concept, "subject": questioner["subject"],
            "assertion": assertion, "assertion_key": questioner["assertion_key"],
            "questioner": questioner, "answerer": answerer,
            "verification": {"requested_sources": 3, "completed_sources": len(results),
                             "source_results": results, "errors": errors,
                             "spectrum": {"minimum": 0, "maximum": 100,
                                          "dots": scores, "average": round(average, 1)}},
            "final": final,
        }
        self._append_ledger(concept, "adjudication", adjudication)
        self.emit("adjudication", record=adjudication)
        return adjudication


def build_assertion_graph(
    concept: str,
    adjudication: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the active assertion graph.

    The graph distinguishes:
    - the concept;
    - the original factual assertion;
    - the verified correction.
    """

    questioner = adjudication.get("questioner", {})
    final = adjudication.get("final", {})

    category = final.get("category")

    if not isinstance(category, dict):
        category = confidence_category(50.0)

    assertion = str(
        adjudication.get("assertion")
        or questioner.get("assertion")
        or questioner.get("asserted_value")
        or "Unknown assertion"
    ).strip()

    corrected_fact = str(
        final.get("corrected_fact")
        or "Verified correction unavailable"
    ).strip()

    try:
        support = float(
            final.get("assertion_support", 50.0)
        )
    except (TypeError, ValueError):
        support = 50.0

    support = max(
        0.0,
        min(100.0, support),
    )

    category_label = str(
        category.get("label")
        or "Uncertain"
    )

    category_colour = str(
        category.get("colour")
        or "#facc15"
    )

    return {
        "nodes": [
            {
                "id": "concept",
                "label": concept,
                "kind": "concept",
                "colour": "#2563eb",
                "note": f"Active concept: {concept}",
            },
            {
                "id": "asserted",
                "label": assertion,
                "kind": "assertion",
                "colour": category_colour,
                "note": (
                    f"{category_label} — "
                    f"assertion support {support:.1f}%"
                ),
            },
            {
                "id": "corrected",
                "label": corrected_fact,
                "kind": "verified",
                "colour": "#22c55e",
                "note": (
                    "Corrected fact supported by the "
                    "completed three-source adjudication."
                ),
            },
        ],
        "edges": [
            {
                "source": "concept",
                "target": "asserted",
                "label": f"asserted — {support:.1f}%",
            },
            {
                "source": "concept",
                "target": "corrected",
                "label": "verified correction",
            },
        ],
    }