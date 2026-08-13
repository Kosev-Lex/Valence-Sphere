# adjudicator.py
# -------------------------------------------------------------
# Adjudicator — impartial Stage 1 answer evaluation controller.
#
# Responsibilities:
#   • Load reasoning and decision rules from adjudicator_rules.json
#   • Evaluate new (Q, A) results for acceptance, pending, or rejection
#   • Compute confidence score (0–100)
#   • Detect contradictions and semantic drift
#   • Route outcomes to SymbolicMemoryEngine
#   • Write JSONL trace logs for review
#
# JL Kosev-Lex, 2025-11-17 (complete edition)
# -------------------------------------------------------------

import os, json, re, math
from datetime import datetime, timezone
from typing import Dict, Any, List


class Adjudicator:
    def __init__(self, memory, rules_path: str = "adjudicator_rules.json"):
        """
        The Adjudicator decides whether new facts are accepted, rejected, or pending.
        It references logical rules, logs each decision, and writes adjudication records
        to ValenceSphere/_adjudication_logs/<date>.jsonl in a reliable, cross-platform way.
        """
        self.memory = memory
        candidate = os.path.abspath(rules_path)
        self.rules_path = candidate if os.path.exists(candidate) else os.path.join(os.path.dirname(__file__), rules_path)
        self.rules = self._load_rules()

        vs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ValenceSphere"))
        self.log_dir = os.path.join(vs_dir, "_adjudication_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"[INIT] Adjudicator log directory → {self.log_dir}")

    # ---------------------------------------------------------
    # RULES
    # ---------------------------------------------------------
    def _load_rules(self) -> Dict[str, Any]:
        """Load hierarchical rules from JSON; fall back to defaults."""
        if not os.path.exists(self.rules_path):
            print(f"[Adjudicator] Warning: rules file not found at {self.rules_path}. Using defaults.")
            return {
                "logic": {"contradiction_check": True, "identity_check": True},
                "credibility": {"source_count": True, "consistency": True},
                "relevance": {"context_match": True},
                "confidence_weights": {"base": 0.6, "coherence_bonus": 0.2, "source_bonus": 0.2}
            }
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Adjudicator] Error loading rules: {e}")
            return {}

    # ---------------------------------------------------------
    # INCOMPLETE ANSWER DETECTION
    # ---------------------------------------------------------
    def is_incomplete_answer(self, text: str) -> bool:
        """Detect truncated or malformed answers."""
        text = (text or "").strip()
        if not text:
            return True

        lowered = text.lower()
        truncation_signals = [
            "incomplete answer", "truncated", "retry recommended",
            "cut off", "unfinished", "continue", "…"
        ]
        if any(sig in lowered for sig in truncation_signals):
            return True
        if lowered.rstrip(".?!") in {"unclear", "unknown", "not sure", "i don't know", "i do not know"}:
            return True
        if lowered.endswith((':', '-', '—', '(', '/', '[')):
            return True
        if re.match(r"^\s*(\d+[\.\)]|[-•*])\s*$", lowered):
            return True
        if len(text.split()) < 2 and not text.endswith("."):
            return True
        return False

    # ---------------------------------------------------------
    # CORE JUDGMENT
    # ---------------------------------------------------------
    def judge(self, question: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a (question, answer) pair under adjudicator rules."""
        qtext = (question.get("text") or "").strip()
        atext = (answer.get("answer_text") or "").strip()
        concept = (question.get("concept") or "unknown").strip()
        role = (question.get("role") or "misc").lower()

        # --- 0. Incomplete answer ---
        if self.is_incomplete_answer(atext):
            verdict = {
                "decision": "pending",
                "confidence": 25.0,
                "reasoning": "Answer appears truncated or incomplete; queued for retry.",
                "related_conflicts": [],
                "timestamp": self._now()
            }
            if hasattr(self.memory, "store_pending"):
                try:
                    self.memory.store_pending(concept, question, answer, verdict)
                except Exception as e:
                    print(f"[ADJ] Error storing pending answer: {e}")
            self._log_adjudication(concept, question, answer, verdict)
            print(f"[ADJ] Incomplete answer detected for {concept}.{role}, held for retry.")
            return verdict

        # --- 1. Base confidence ---
        quality_score = answer.get("evaluation", {}).get("score")
        conf = self._estimate_confidence(atext, quality_score)

        # --- 2. Conflict detection ---
        conflicts = self._check_conflicts(concept, atext)
        if conflicts:
            self._log_conflicts(concept, conflicts)

        # --- 3. Context drift detection (transparent lexical heuristic) ---
        drift_detected, branch_created, context_reason = False, None, None
        try:
            base_def = getattr(self.memory, "get_fact_text", lambda c, r: None)(concept, "what")
            if base_def:
                base_tokens = set(re.findall(r"[a-z]{3,}", base_def.casefold()))
                answer_tokens = set(re.findall(r"[a-z]{3,}", atext.casefold()))
                overlap = len(base_tokens & answer_tokens) / max(1, len(base_tokens | answer_tokens))
                # Only branch when the answer also contains an explicit alternate-domain marker.
                markers = {
                    "company": "Company", "corporation": "Company", "brand": "Company",
                    "color": "Color", "colour": "Colour", "shade": "Color",
                }
                marker = next((label for word, label in markers.items() if word in answer_tokens), None)
                if overlap < 0.08 and marker:
                    drift_detected = True
                    branch_created = f"{concept}: {marker}"
                    context_reason = f"Detected lexical domain drift (overlap={overlap:.2f})."
        except Exception as e:
            print(f"[ADJ] Context drift check error: {e}")

        # ==========================================================
        # === 4. Decision logic (rule-driven thresholds) ===========
        # ==========================================================
        rule_blocks = self.rules.get("rules", self.rules)
        disp_rules = rule_blocks.get("final_disposition", {})
        accept_th = disp_rules.get("accept_threshold", 0.75) * 100
        pending_th = disp_rules.get("pending_threshold", 0.40) * 100
        reject_th = disp_rules.get("reject_threshold", 0.20) * 100
        default_disp = disp_rules.get("default_disposition", "pending")

        if drift_detected and branch_created:
            decision = "pending_divergent"
            reasoning = f"{context_reason} Suggest spawning branch concept '{branch_created}'."
            conf = max(conf, accept_th * 0.8)
        elif conflicts:
            decision = "pending"
            reasoning = f"Conflict detected with {len(conflicts)} accepted fact(s)."
        elif conf >= accept_th:
            decision = "accepted"
            reasoning = "Confidence meets acceptance threshold; coherent and conflict-free."
        elif conf >= pending_th:
            decision = "pending"
            reasoning = "Confidence is moderate; queued for review."
        elif conf <= reject_th:
            decision = "rejected"
            reasoning = "Confidence below rejection threshold; insufficient coherence."
        else:
            decision = default_disp
            reasoning = f"Default disposition applied ({default_disp})."

        # --- 5. Verdict ---
        verdict = {
            "decision": decision,
            "confidence": round(conf, 1),
            "reasoning": reasoning,
            "related_conflicts": conflicts,
            "timestamp": self._now(),
            "provenance": {
                "question_id": question.get("id"),
                "source_model": answer.get("model", "unknown"),
                "timestamp": self._now()
            }
        }
        if branch_created:
            verdict["branch_created"] = branch_created

        # --- 6. Memory routing ---
        try:
            if decision == "accepted" and hasattr(self.memory, "store_accepted"):
                self.memory.store_accepted(concept, question, answer, verdict)
            elif decision == "pending" and hasattr(self.memory, "store_pending"):
                self.memory.store_pending(concept, question, answer, verdict)
            elif decision == "rejected" and hasattr(self.memory, "log_rejection"):
                self.memory.log_rejection(concept, question, answer, verdict)
        except Exception as e:
            print(f"[ADJ] Memory routing error: {e}")

        # --- 7. Logging ---
        self._log_adjudication(concept, question, answer, verdict)

        if drift_detected:
            print(f"[ADJ] Context drift detected for {concept}.{role} → {branch_created}")
        else:
            print(f"[ADJ] Adjudicated {concept}.{role}: {decision} ({conf:.1f}%)")

        return verdict

    # ---------------------------------------------------------
    # CONFLICT DETECTION
    # ---------------------------------------------------------
    def _check_conflicts(self, concept: str, new_text: str) -> List[Dict[str, str]]:
        """Find contradictions with previously accepted facts."""
        conflicts = []
        try:
            node = getattr(self.memory, "ensure_concept", lambda c: None)(concept)
            if not node or not hasattr(node, "roles"):
                return []
            for role, slot in (node.roles or {}).items():
                if not hasattr(slot, "claims"):
                    continue
                for claim in getattr(slot, "claims", []):
                    ctext = (getattr(claim, "text", "") or "").strip().lower()
                    if ctext and self._is_contradictory(ctext, new_text.lower()):
                        conflicts.append({"role": role, "text": ctext})
        except Exception as e:
            print(f"[Adjudicator] conflict check error: {e}")
        return conflicts

    def _is_contradictory(self, old: str, new: str) -> bool:
        """Simple contradiction heuristic."""
        if " not " in old and " not " not in new and old.replace(" not ", " ") in new:
            return True
        if " not " in new and " not " not in old and new.replace(" not ", " ") in old:
            return True
        opposites = [("fruit", "vegetable"), ("true", "false"), ("yes", "no")]
        return any((a in old and b in new) or (b in old and a in new) for a, b in opposites)

    # ---------------------------------------------------------
    # TRANSPARENT QUALITY ESTIMATION
    # ---------------------------------------------------------
    def _estimate_confidence(self, text: str, quality_score: int | float | None = None) -> float:
        """Estimate answer quality without pretending to verify factual truth."""
        if not text:
            return 0.0
        words = text.split()
        word_count = len(words)
        if word_count < 3:
            length_quality = 0.1
        elif word_count < 5:
            length_quality = 0.6
        elif word_count <= 80:
            length_quality = 1.0
        elif word_count <= 120:
            length_quality = 0.7
        else:
            length_quality = 0.4
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
        sentence_quality = 1.0 if 1 <= sentence_count <= 3 else 0.5
        completeness = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.2
        try:
            answer_quality = min(1.0, max(0.0, float(quality_score) / 5.0))
        except (TypeError, ValueError):
            answer_quality = 0.6
        lowered = text.casefold()
        hedge_terms = ("might", "possibly", "perhaps", "not sure", "i think", "unclear")
        hedge_penalty = 0.30 if any(term in lowered for term in hedge_terms) else 0.0
        confidence = (
            0.25
            + 0.30 * answer_quality
            + 0.20 * length_quality
            + 0.15 * sentence_quality
            + 0.10 * completeness
            - hedge_penalty
        )
        return round(min(1.0, max(0.0, confidence)) * 100, 1)

    # ---------------------------------------------------------
    # LOGGING
    # ---------------------------------------------------------
    def _log_adjudication(self, concept: str, q: Dict[str, Any], a: Dict[str, Any], verdict: Dict[str, Any]):
        """Append adjudication record to a dated JSONL file."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = os.path.join(self.log_dir, f"{date_str}.jsonl")
            rec = {
                "timestamp": self._now(),
                "concept": concept,
                "question": q.get("text", ""),
                "answer": a.get("answer_text", ""),
                "decision": verdict.get("decision"),
                "confidence": verdict.get("confidence"),
                "reasoning": verdict.get("reasoning"),
                "related_conflicts": verdict.get("related_conflicts", []),
                "provenance": verdict.get("provenance", {})
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Adjudicator] log error: {e}")

    def _log_conflicts(self, concept: str, conflicts: List[Dict[str, str]]):
        """Separate conflict log for analytical review."""
        if not conflicts:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            path = os.path.join(self.log_dir, f"conflicts_{date_str}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                for c in conflicts:
                    f.write(json.dumps({
                        "timestamp": self._now(),
                        "concept": concept,
                        "role": c.get("role"),
                        "text": c.get("text")
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Adjudicator] conflict log error: {e}")

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
