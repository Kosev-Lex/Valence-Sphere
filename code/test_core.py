import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import concept_repository as repo
import template_schema
from adjudicator import Adjudicator
from factcheck_engine import FactCheckEngine, build_assertion_graph, confidence_category
from knowledge_graph import build_concept_graph
from llm_services import ConstituentAnalysisService, LLMResponse
from model_stage import APIProfileStore, AgentAPIClient
from plato import Answerer
from socrates import Questioner


class FakeSource:
    def __init__(self, response="A concise answer."):
        self.request = ""
        self.response = response

    def fetch(self, question_text, endpoint, params):
        self.request = question_text
        return {"raw_text": self.response, "provider": "fake", "endpoint": endpoint, "usage": {}}


class FakeConstituentLLM:
    def complete(self, system, user, temperature=None):
        return LLMResponse(
            text=json.dumps({"core_meaning": "A cat is a mammal", "nouns": ["cat", "mammal"],
                             "verbs": ["is"], "adjectives": [], "adverbs": [],
                             "noun_phrases": ["a cat", "a mammal"], "verb_phrases": ["is a mammal"],
                             "qualifiers": [], "relations": ["cat is mammal"]}),
            model="fake", usage={"total_tokens": 1})


class FakeFactClient:
    calls = []

    def __init__(self, role):
        self.role = role
        self.profile = {"model": f"fake-{role}", "native_web_search": False}

    def complete(self, system, user, attachments):
        type(self).calls.append(self.role)
        if self.role == "questioner":
            payload = [{"subject": "lemon colour", "assertion": "Lemons are red",
                        "question": "Are lemons red?", "concept_hint": "lemon",
                        "asserted_value": "red"}]
        elif self.role == "answerer":
            payload = {"relationship": "contradicts", "direct_answer": "No.",
                       "assertion_support": 0, "reasoning": "The template says yellow.",
                       "corrected_fact": "Lemons are yellow.", "flag": "incorrect"}
        elif self.role.startswith("source_"):
            score = {"source_1": 0, "source_2": 5, "source_3": 10}[self.role]
            payload = {"position_score": score, "confidence": 95, "verdict": "contradicts",
                       "explanation": "The assertion is contradicted.",
                       "corrected_fact": "Lemons are yellow.",
                       "sources": [{"title": f"Evidence {self.role[-1]}",
                                    "url": f"https://example.test/{self.role}"}]}
        else:
            payload = {"summary": "The assertion is contradicted by all three checks.",
                       "corrected_fact": "Lemons are yellow.",
                       "note": "Three-source result."}
        return LLMResponse(text=json.dumps(payload), model=f"fake-{self.role}",
                           usage={"total_tokens": 1})


def lemon_record():
    data = {"concept": "Lemon", "discovery": {"questions": [
        {"id": 1, "text": "What is Lemon?", "phase": "definition", "status": "answered",
         "answer": "Lemon is a citrus fruit known for its bright yellow color.",
         "confidence": 100.0},
        {"id": 2, "text": "What colour is Lemon?", "phase": "properties", "status": "answered",
         "answer": "Lemon is a citrus fruit known for its bright yellow color.",
         "confidence": 98.0},
    ]}}
    return repo.ConceptRecord("Lemon", Path("lemon"), Path("lemon/lemon.json"), None, data)


class CoreTests(unittest.TestCase):
    def test_native_concept_path_is_one_directory_and_one_template(self):
        self.assertEqual(repo.concept_path("Lemon").parts[-2:], ("lemon", "lemon.json"))
        self.assertEqual(repo.audit_path("Lemon", "socrates").name, "socrates_lemon.json")

    def test_fresh_template_has_exactly_twenty_questions(self):
        template = template_schema.make_blank_template("Lemon")
        self.assertEqual(template["version"], "11.0")
        self.assertEqual(len(template["discovery"]["questions"]), 20)
        self.assertEqual(set(template["discovery"]), {"question_bank_version", "questions"})

    def test_atomic_update_keeps_one_canonical_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            repo.atomic_write_json(path, {"revision": 1})
            repo.atomic_write_json(path, {"revision": 2})
            self.assertEqual(json.loads(path.read_text()), {"revision": 2})
            self.assertEqual(list(Path(tmp).glob("*.json")), [path])

    def test_constituents_update_the_same_concept_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lemon.json"
            repo.atomic_write_json(path, {"concept": "Lemon", "discovery": {"questions": [
                {"id": 1, "text": "What is Lemon?"}]}})
            repo.save_question_constituents(path, 1, {"result": {"nouns": ["lemon"]}})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["discovery"]["questions"][0]
                             ["constituent_analysis"]["result"]["nouns"], ["lemon"])
            self.assertEqual([item.name for item in Path(tmp).glob("*.json")], ["lemon.json"])

    def test_learning_save_retains_concurrent_constituents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lemon.json"
            repo.atomic_write_json(path, {"concept": "Lemon", "discovery": {"questions": [
                {"id": 1, "answer": "A citrus fruit.",
                 "constituent_analysis": {"result": {"nouns": ["fruit"]}}}]}})
            repo.save_concept_template(path, {"concept": "Lemon", "discovery": {"questions": [
                {"id": 1, "answer": "A citrus fruit."}]}})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["discovery"]["questions"][0]
                             ["constituent_analysis"]["result"]["nouns"], ["fruit"])

    def test_questioner_labels_definition_as_context(self):
        template = {"discovery": {"questions": [
            {"id": 1, "text": "What is a cat?", "status": "answered", "answer": "A small mammal."},
            {"id": 2, "text": "Where is a cat found?", "status": "unasked"}]}}
        question = Questioner("cat").next_question(template)
        self.assertEqual(question["context"]["text"], "A small mammal.")
        source = FakeSource()
        Answerer(source).answer(question)
        self.assertIn("QUESTION TO ANSWER", source.request)
        self.assertIn("CONTEXT FROM THE ACCEPTED DEFINITION", source.request)

    def test_constituent_service_parses_semantic_categories(self):
        analysis, response = ConstituentAnalysisService(FakeConstituentLLM()).analyze(
            "What is a cat?", "A mammal.")
        self.assertEqual(analysis["nouns"], ["cat", "mammal"])
        self.assertEqual(response.model, "fake")

    def test_answerer_compression_keeps_hyphenated_words(self):
        source = FakeSource("One fact. A well-being fact. A third fact. A discarded fourth fact.")
        answer = Answerer(source).answer({"id": 1, "text": "Explain it."})
        self.assertIn("well-being", answer["answer_text"])
        self.assertNotIn("fourth", answer["answer_text"])

    def test_adjudicator_confidence_distinguishes_hedging(self):
        adjudicator = Adjudicator(memory=object())
        solid = adjudicator._estimate_confidence("A cat is a small domesticated mammal.", 5)
        hedged = adjudicator._estimate_confidence("This might possibly be true.", 5)
        self.assertGreater(solid, hedged)

    def test_concept_graph_contains_parent_edge(self):
        parent = repo.ConceptRecord("animal", Path("animal"), Path("animal/animal.json"), None,
                                    {"concept": "animal", "discovery": {"questions": []}})
        child = repo.ConceptRecord("cat", Path("cat"), Path("cat/cat.json"), "animal",
                                   {"concept": "cat", "discovery": {"questions": []}})
        graph = build_concept_graph([parent, child])
        self.assertEqual({node["name"] for node in graph["nodes"]}, {"animal", "cat"})
        self.assertTrue(any(edge["label"] == "parent" for edge in graph["edges"]))

    def test_full_factcheck_writes_cumulative_ledgers_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {stage: root / f"{stage}_lemon.json"
                     for stage in ("socrates", "answerer", "adjudication")}
            FakeFactClient.calls = []
            engine = FactCheckEngine(lambda role: FakeFactClient(role))
            with patch("factcheck_engine.discover_concepts", return_value=[lemon_record()]), \
                 patch("factcheck_engine.audit_path", side_effect=lambda _concept, stage: paths[stage]):
                first = engine.scan_message("Lemons are red", "user", "m1")
                second = engine.scan_message("Lemons are red", "user", "m2")
            self.assertFalse(first[0].cached)
            self.assertTrue(second[0].cached)
            self.assertEqual(first[0].adjudication["verification"]["spectrum"]["dots"], [0.0, 5.0, 10.0])
            self.assertEqual(first[0].adjudication["final"]["category"]["key"], "incorrect")
            self.assertEqual(FakeFactClient.calls.count("source_1"), 1)
            self.assertEqual(len(json.loads(paths["socrates"].read_text())["records"]), 2)
            self.assertEqual(len(json.loads(paths["answerer"].read_text())["records"]), 2)
            self.assertEqual(len(json.loads(paths["adjudication"].read_text())["records"]), 1)

    def test_evidence_deduplicates_and_counts_repeated_entry(self):
        claim = {"subject": "lemon colour", "assertion": "Lemons are red",
                 "question": "Are lemons red?"}
        evidence = FactCheckEngine._retrieve_evidence(lemon_record().data, claim)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["occurrences"], 2)
        self.assertEqual(evidence[0]["confidence"], 100.0)

    def test_confidence_colours_and_assertion_graph(self):
        self.assertEqual(confidence_category(5)["colour"], "#ef4444")
        self.assertEqual(confidence_category(50)["colour"], "#facc15")
        self.assertEqual(confidence_category(90)["colour"], "#22c55e")
        adjudication = {"assertion": "Lemons are red", "questioner": {"asserted_value": "red"},
                        "final": {"assertion_support": 5, "category": confidence_category(5),
                                  "corrected_fact": "Lemons are yellow."}}
        graph = build_assertion_graph("Lemon", adjudication)
        self.assertEqual({node["kind"] for node in graph["nodes"]},
                         {"concept", "assertion", "verified"})

    def test_agent_profiles_save_key_names_not_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profiles.json"
            with patch("model_stage.PROFILE_PATH", profile_path):
                store = APIProfileStore()
                values = store.defaults()
                values["questioner"]["api_key_env"] = "QUESTIONER_API_KEY"
                keys = {role: "secret-value" for role in values}
                store.update_all(values, keys)
                saved = json.loads(profile_path.read_text())
                self.assertEqual(saved["questioner"]["api_key_env"], "QUESTIONER_API_KEY")
                self.assertNotIn("api_key", saved["questioner"])
                self.assertNotIn("secret-value", profile_path.read_text())

    def test_agent_attachments_include_text_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            note, image = Path(tmp) / "note.txt", Path(tmp) / "reference.png"
            note.write_text("important attached context", encoding="utf-8")
            image.write_bytes(b"image-bytes")
            content = AgentAPIClient._content("User request", [note, image])
            self.assertIsInstance(content, list)
            self.assertIn("important attached context", content[0]["text"])
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
