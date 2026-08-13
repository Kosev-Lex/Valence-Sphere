# plato.py
# aka answerer module
# Answerer is distinct from InformationSource.
# Answerer prepares request, calls InformationSource, extracts a precise answer, and returns it.

import os, re, json, uuid
from datetime import datetime, timezone
from typing import Dict, Any
from constants import DEFAULT_OPENAI_MODEL
from text_quality import clean_answer_text, is_incomplete


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# Information Source (API adapter)
# ----------------------------

# --- Strict ValenceSphere Answerer prompt ---
SYSTEM_PROMPT_VS_ANSWERER = """

Follow these strict formatting and style rules for every answer:

1. Respond in plain text only, as one continuous block.
2. Provide one to three short sentences — never lists, outlines, or enumerations.
3. Never use colons or semicolons to begin a list or clause unless the sentence is grammatically complete.
   - Do not end any sentence with a colon or semicolon.
   - Do not write "1.", "2.", "Firstly", or any bullet-like pattern.
4. Do not include introductions ("Answer:", "The answer is") or conclusions ("Therefore", "In summary").
5. Maintain a neutral, confident, factual tone. Avoid speculation, opinion, or filler.
6. Do not use markdown, bullets, emojis, or multiple paragraphs.
7. Keep everything concise and declarative — one continuous sentence block.

If you receive a question asking for reasons, causes, or factors, compress the explanation into a single fluent sentence using “because”, “due to”, or “as a result of”, without colons or numbered lists.

Example — good:
"Tomatoes became commercial because they are easy to cultivate, transport, and store."

Example — bad:
"Tomatoes became commercial primarily due to several key factors: 1. High demand 2. Shelf life."
"""


class InformationSource:
    """
    Handles the low-level API call for fetching answers.
    Returns a consistent dictionary with provider info, endpoint, params, and usage stats.
    """
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, temperature: float = 0.2):
        self.model = model or DEFAULT_OPENAI_MODEL
        self.temperature = temperature
        self.api_key = os.getenv("VS_API") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.client = None
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use API learning.") from exc
        client_args = {"api_key": self.api_key}
        if os.getenv("VS_BASE_URL"):
            client_args["base_url"] = os.environ["VS_BASE_URL"]
        self.client = OpenAI(**client_args)

    def fetch(self, question_text: str, endpoint: str = "openai.chat.completions", params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calls the OpenAI Chat Completions API and returns a normalized structure.
        Automatically injects the strict ValenceSphere Answerer system prompt.
        """
        params = (params or {}).copy()
        model = params.get("model") or self.model or DEFAULT_OPENAI_MODEL
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", 200)

        try:
            if self.client is None:
                raise EnvironmentError("Set VS_API or OPENAI_API_KEY before starting API learning.")
            # Inject the strict Answerer protocol
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_VS_ANSWERER.strip()},
                    {"role": "user", "content": question_text.strip()},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            message = response.choices[0].message
            text = getattr(message, "content", "").strip()
            usage = getattr(response, "usage", None)
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            } if usage else {"total_tokens": 0}

            return {
                "provider": "openai",
                "endpoint": endpoint,
                "params": {"model": model, "temperature": temperature, "max_tokens": max_tokens},
                "raw_text": text,
                "usage": usage_dict,
            }

        except Exception as e:
            return {
                "provider": "openai",
                "endpoint": endpoint,
                "params": {"model": model, "temperature": temperature, "max_tokens": max_tokens},
                "raw_text": "",
                "error": str(e),
                "usage": {"total_tokens": 0},
            }


# ----------------------------
# Answerer (distinct from API)
# ----------------------------
class Answerer:
    def __init__(self, info_source, instruction_profile_path: str = "gpt_instruction_profile.json"):
        self.src = info_source
        self.instruction_profile_path = instruction_profile_path
        self.api_key = os.getenv("VS_API") or os.getenv("OPENAI_API_KEY")
        self._profile = self._load_instruction_profile()

    # ------------------------
    # Core Answering Pipeline
    # ------------------------
    def answer(self, qobj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an answer for a given question object (qobj).
        Ensures 'prep' exists, calls the provider, sanitizes text,
        and flags incomplete or truncated responses.
        """
        import uuid

        # --- Safety guard: ensure prep exists and is consistent ---
        if "prep" not in qobj:
            qobj["prep"] = {
                "endpoint": "openai.chat.completions",
                "params": {"model": DEFAULT_OPENAI_MODEL, "temperature": 0.2},
            }

        endpoint = qobj["prep"].get("endpoint", "openai.chat.completions")
        params = qobj["prep"].get("params", {"model": DEFAULT_OPENAI_MODEL, "temperature": 0.2})

        # --- Fetch raw answer from provider ---
        question_text = qobj.get("text", "")
        context = qobj.get("context") or {}
        if context.get("text"):
            request_text = (
                f"QUESTION TO ANSWER:\n{question_text}\n\n"
                "CONTEXT FROM THE ACCEPTED DEFINITION:\n"
                f"{context['text']}\n\n"
                "The context is background only. Answer the QUESTION TO ANSWER directly; "
                "do not repeat or treat the context as the requested answer."
            )
        else:
            request_text = question_text
        raw = self.src.fetch(request_text, endpoint, params)

        # --- Extract and sanitize ---
        cleaned = self._extract_precise_answer(raw) or {}
        raw_text = cleaned.get("text", "") or ""
        ans_text = self._sanitize(raw_text)

        # --- Handle incomplete or truncated responses ---
        if self._is_incomplete(ans_text):
            ans_text = f"{ans_text} (incomplete answer truncated; retry recommended)"

        # --- Score the cleaned answer ---
        score = self._evaluate_answer(ans_text)

        # --- Provider metadata ---
        provider = raw.get("provider", "unknown")
        endpoint_name = raw.get("endpoint", endpoint)
        usage = raw.get("usage", {}) or {}

        # --- Update cleaned payload ---
        cleaned["text"] = ans_text

        # --- Structured answer object ---
        return {
            "question_id": qobj.get("id", str(uuid.uuid4())),
            "question_text": qobj.get("text", ""),
            "context_used": context.get("text", ""),
            "model": raw.get("params", {}).get("model", params.get("model", DEFAULT_OPENAI_MODEL)),
            "tier": qobj.get("tier", 1),
            "role": qobj.get("role", "misc"),
            "answer_text": ans_text,
            "evidence": cleaned.get("evidence", {}),
            "evaluation": {
                "score": score,
                "criteria": self._profile.get("response_evaluation", {}).get("criteria", {}),
            },
            "provider_payload": {
                "provider": provider,
                "endpoint": endpoint_name,
                "meta": usage,
            },
            "timestamp": now_iso(),
            "error": raw.get("error"),
        }

    def _is_incomplete(self, text: str) -> bool:
        """Detect truncated or incomplete answers."""
        if not text:
            return True
        text = text.strip().lower()
        if text.endswith((':', ';', '-', '—', '(')):
            return True
        if re.search(r"(incomplete|truncated|retry recommended)", text):
            return True
        if len(text.split()) < 3 and not text.endswith('.'):
            return True
        return False

    # ------------------------
    # Extraction Logic
    # ------------------------
    # https://github.com/Kosev-Lex
    def _extract_precise_answer(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the clean factual sentence from model output.
        Enforces plain-text style consistent with ValenceSphere protocol.
        """
        text = (raw.get("raw_text") or "").strip()
        provider = raw.get("provider", "unknown")
        if not text:
            return {"text": "", "evidence": {"quote": "", "source": provider}}

        # Preserve up to the three sentences permitted by the Answerer prompt.
        # Keeping only sentence one silently discarded useful content on broad
        # questions such as structure, mechanism, and impact.
        sentence = " ".join(re.split(r"(?<=[.!?])\s+", text)[:3])
        sentence = self._sanitize(sentence)

        return {
            "text": sentence,
            "evidence": {"quote": sentence, "source": provider}
        }

    def _sanitize(self, text: str) -> str:
        """
        Clean and normalize model output before adjudication.
        Removes markdown, emojis, filler phrases, and incomplete trailing punctuation.
        Ensures final output ends as a complete sentence.
        """
        if not text:
            return ""

        # --- Strip markdown / symbols / bullets ---
        text = re.sub(r"[*_`~>|#•●▪️]", "", text)
        text = re.sub(r"(?m)^\s*[-+]\s+", "", text)

        # --- Remove emojis and emoji-like characters ---
        text = re.sub(r"[🙂🙃😊😅😂😉🤔🤷‍♂️🤷‍♀️🙌❤️💡🔥✨👍👎🙏💭🧠]", "", text)

        # --- Remove common lead-ins like "Answer:", "Sure!", "Here's" ---
        text = re.sub(
            r"^(Answer:|Sure!|Certainly!|Of course!|The answer is|Here’s|Here's|Let's see|Sure, )\s*",
            "",
            text,
            flags=re.I,
        )

        # --- Clean common closing truncation markers ---
        text = re.sub(
            r"\(incomplete answer.*?retry.*?\)", "",
            text,
            flags=re.I
        )

        # --- Remove dangling punctuation or filler endings ---
        text = text.strip()
        text = re.sub(r"[:;,\-–—]+$", "", text).strip()

        # --- Replace excessive whitespace ---
        text = re.sub(r"\s{2,}", " ", text)

        # --- Ensure final punctuation is present ---
        if text and not text.endswith(('.', '!', '?')):
            text += '.'

        return text.strip()

    # ------------------------
    # GPT Profile Loader
    # ------------------------
    def _load_instruction_profile(self) -> Dict[str, Any]:
        """Loads GPT instruction schema; falls back to a minimal directive."""
        candidate = self.instruction_profile_path
        if not os.path.isabs(candidate) and not os.path.exists(candidate):
            candidate = os.path.join(os.path.dirname(__file__), candidate)
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # Fallback minimal schema
        return {
            "name": "VS_Minimal_Profile",
            "directives": {
                "output_format": "Plain text only. No markdown or formatting.",
                "style": "Succinct and factual.",
                "scope": "Answer only the specific question asked."
            },
            "response_evaluation": {
                "criteria": {
                    "precision": "Directly answers the question",
                    "brevity": "Under three sentences",
                    "objectivity": "Free from opinion or filler",
                    "clarity": "Clear and unambiguous",
                    "structure": "Single coherent text block"
                },
                "scoring_scale": {
                    "1": "Off-topic or verbose",
                    "2": "Irrelevant or redundant info",
                    "3": "Acceptable but not concise",
                    "4": "Concise and factual",
                    "5": "Ideal — precise, succinct, text-only answer"
                }
            }
        }

    # ------------------------
    # Evaluation Logic
    # ------------------------
    def _evaluate_answer(self, text: str) -> int:
        """Simple heuristic 1–5 scoring of precision, brevity, objectivity, clarity, structure."""
        if not text or len(text.strip()) < 3:
            return 1
        word_count = len(text.split())
        sentence_count = len(re.findall(r"[.!?]", text))
        score = 5
        if word_count > 50:
            score -= 1
        if sentence_count > 3:
            score -= 1
        if re.search(r"\b(I|we|my|our|me)\b", text, re.I):
            score -= 1
        if re.search(r"[*#\-•]", text):
            score -= 1
        return max(1, min(score, 5))

    # ------------------------
    # Direct API Call (if needed)
    # ------------------------

    def fetch_via_openai(self, question_text: str, model: str = "gpt-4o-mini") -> str:
        """
        Fetches an answer from GPT under the VS_Answerer_Protocol_v2 profile.
        Enforces factual, list-free, direct, single-block responses.
        """
        if not self.api_key:
            raise EnvironmentError("Missing VS_API or OPENAI_API_KEY.")

        if self.src.client is None:
            raise EnvironmentError("Missing VS_API or OPENAI_API_KEY.")

        # --- Load instruction profile ---
        profile = self._profile or {}
        name = profile.get("name", "VS Answerer Protocol")
        purpose = profile.get("purpose", "Provide concise, factual answers.")
        directives = profile.get("directives", {})
        rules = json.dumps(directives, indent=2)

        # --- Construct System Prompt ---
        system_prompt = (
            f"You are a factual answering module operating under the {name}.\n"
            f"Purpose: {purpose}\n"
            f"Follow these rules strictly:\n{rules}\n\n"
            f"Always reply in a single, continuous sentence block.\n"
            f"Never enumerate, never restate the question, and never introduce or conclude."
        )

        # --- Compose User Prompt ---
        user_prompt = (
            f"Question: {question_text}\n\n"
            f"Answer directly, using plain language as if explaining to a young learner.\n"
            f"Do not use bullets, numbering, or multiple sentences unless absolutely necessary."
        )

        # --- Call GPT API ---
        try:
            response = self.src.client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()

            # --- Post-clean to remove any stray list or formatting remnants ---
            answer = re.sub(r"^[\d\-\*\•]+\s*", "", answer)
            answer = re.sub(r"\n+", " ", answer).strip()
            if answer.endswith(":"):
                answer += " (incomplete answer truncated; retry recommended)"
            return answer

        except Exception as e:
            return f"[API Error: {e}]"
