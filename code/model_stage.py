"""Stage 2 ordinary chat with asynchronous ValenceSphere fact checking."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from concept_repository import GLOBAL_DIR, ROOT_DIR, atomic_write_json, discover_concepts, load_json, safe_filename
from factcheck_engine import FactCheckEngine, FactCheckOutcome, build_assertion_graph, confidence_category
from llm_services import LLMResponse, LLMServiceError


WORKSPACE_ROOT = ROOT_DIR / "_model_workspaces"
PROFILE_PATH = GLOBAL_DIR / "model_api_profiles.json"
ROLES = ("chat", "questioner", "answerer", "adjudicator", "source_1", "source_2", "source_3")
ROLE_LABELS = {
    "chat": "Chat", "questioner": "Questioner", "answerer": "Answerer",
    "adjudicator": "Adjudicator", "source_1": "Source 1",
    "source_2": "Source 2", "source_3": "Source 3",
}
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".py",
                 ".html", ".css", ".js", ".xml", ".yaml", ".yml"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class APIProfileStore:
    """Independent chat, triad and verification-source configurations."""

    SAVED_FIELDS = {"provider", "model", "base_url", "api_key_env", "temperature",
                    "native_web_search"}

    @staticmethod
    def defaults() -> dict[str, dict[str, Any]]:
        profiles = {}
        for role in ROLES:
            profiles[role] = {
                "provider": "OpenAI-compatible",
                "model": os.getenv("VS_MODEL", "gpt-4o-mini"),
                "base_url": os.getenv("VS_BASE_URL", ""),
                "api_key_env": "VS_API",
                "temperature": 0.5 if role == "chat" else 0.0,
                "native_web_search": False,
            }
        return profiles

    def __init__(self):
        self.profiles = self.defaults()
        self.session_keys = {role: "" for role in ROLES}
        saved = load_json(PROFILE_PATH, {})
        if isinstance(saved, dict):
            for role in ROLES:
                if isinstance(saved.get(role), dict):
                    self.profiles[role].update({key: value for key, value in saved[role].items()
                                               if key in self.SAVED_FIELDS})

    def get(self, role: str) -> dict[str, Any]:
        profile = dict(self.profiles[role])
        profile["_session_api_key"] = self.session_keys[role]
        return profile

    def update_all(self, values: dict[str, dict[str, Any]], session_keys: dict[str, str]):
        for role in ROLES:
            self.profiles[role].update({key: value for key, value in values[role].items()
                                       if key in self.SAVED_FIELDS})
            self.session_keys[role] = session_keys[role].strip()
        atomic_write_json(PROFILE_PATH, self.profiles)


class AgentAPIClient:
    """OpenAI-compatible transport for one configured role."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = dict(profile)
        env_name = str(self.profile.get("api_key_env") or "VS_API").strip()
        key = (str(self.profile.get("_session_api_key") or "").strip()
               or os.getenv(env_name) or os.getenv("VS_API") or os.getenv("OPENAI_API_KEY"))
        if not key:
            raise LLMServiceError(f"No API key found for {env_name}.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMServiceError("Install the openai package before using Stage 2.") from exc
        kwargs: dict[str, Any] = {"api_key": key}
        if str(self.profile.get("base_url") or "").strip():
            kwargs["base_url"] = str(self.profile["base_url"]).strip()
        self.client = OpenAI(**kwargs)

    @staticmethod
    def _content(text: str, attachments: list[Path]):
        text_parts, images = [text], []
        for path in attachments:
            if not path.exists() or not path.is_file():
                continue
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if mime in IMAGE_TYPES:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                images.append({"type": "image_url",
                               "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            elif path.suffix.casefold() in TEXT_SUFFIXES:
                body = path.read_text(encoding="utf-8", errors="replace")[:100_000]
                text_parts.append(f"\nATTACHED TEXT — {path.name}:\n{body}")
            else:
                text_parts.append(f"\nATTACHED FILE: {path.name} ({mime})")
        combined = "\n".join(text_parts)
        return combined if not images else [{"type": "text", "text": combined}, *images]

    def complete(self, system: str, user: str, attachments: list[Path]) -> LLMResponse:
        if self.profile.get("native_web_search"):
            system += ("\nNative web search/citations are requested when your provider supports them. "
                       "Never invent a URL when search is unavailable.")
        return self._request([
            {"role": "system", "content": system},
            {"role": "user", "content": self._content(user, attachments)},
        ])

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        prepared = [{"role": "system", "content":
                     "You are a normal conversational assistant. Respond naturally to the user. "
                     "Do not mention ValenceSphere, audits, templates, or background fact checking."}]
        prepared.extend({"role": item["role"], "content": item["content"]} for item in messages)
        return self._request(prepared)

    def _request(self, messages: list[dict[str, Any]]) -> LLMResponse:
        model = str(self.profile.get("model") or "gpt-4o-mini")

        request_args = {
            "model": model,
            "messages": messages,
        }

        configured_temperature = self.profile.get("temperature")

        if configured_temperature not in (None, ""):
            request_args["temperature"] = float(configured_temperature)

        try:
            response = self.client.chat.completions.create(**request_args)

        except Exception as exc:
            error_message = str(exc)
            lowered_error = error_message.casefold()

            temperature_rejected = (
                    "temperature" in request_args
                    and "temperature" in lowered_error
                    and any(
                phrase in lowered_error
                for phrase in (
                    "unsupported value",
                    "unsupported parameter",
                    "does not support",
                    "only the default",
                    "not supported",
                )
            )
            )

            if not temperature_rejected:
                raise LLMServiceError(error_message) from exc

            # Some models require their internal default temperature.
            # Retry without sending the temperature parameter.
            request_args.pop("temperature", None)

            try:
                response = self.client.chat.completions.create(**request_args)
            except Exception as retry_exc:
                raise LLMServiceError(str(retry_exc)) from retry_exc

        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=(response.choices[0].message.content or "").strip(),
            model=model,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
        )

class APIConfigurationDialog(tk.Toplevel):
    def __init__(self, owner, store: APIProfileStore, initial_role="chat",
                 on_saved: Callable[[], None] | None = None):
        super().__init__(owner)
        self.title("Stage 2 API Configurations")
        self.geometry("760x610")
        self.transient(owner)
        self.grab_set()
        self.store, self.on_saved = store, on_saved
        self.variables = {}
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        selected = 0
        for index, role in enumerate(ROLES):
            frame = ttk.Frame(notebook, padding=14)
            notebook.add(frame, text=ROLE_LABELS[role])
            if role == initial_role:
                selected = index
            profile = store.get(role)
            values = {
                "provider": tk.StringVar(value=profile["provider"]),
                "model": tk.StringVar(value=profile["model"]),
                "base_url": tk.StringVar(value=profile["base_url"]),
                "api_key_env": tk.StringVar(value=profile["api_key_env"]),
                "api_key": tk.StringVar(value=store.session_keys[role]),
                "temperature": tk.StringVar(value=str(profile["temperature"])),
                "native_web_search": tk.BooleanVar(value=bool(profile.get("native_web_search"))),
            }
            self.variables[role] = values
            frame.columnconfigure(1, weight=1)
            fields = (("Provider label", "provider"), ("Model", "model"),
                      ("Base URL", "base_url"), ("API-key environment variable", "api_key_env"),
                      ("API key (session only)", "api_key"), ("Temperature", "temperature"))
            for row, (label, key) in enumerate(fields):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=8)
                ttk.Entry(frame, textvariable=values[key], show="*" if key == "api_key" else "").grid(
                    row=row, column=1, sticky="we", padx=(12, 0), pady=8)
            ttk.Checkbutton(frame, text="Request native web search and source citations when supported",
                            variable=values["native_web_search"]).grid(
                row=len(fields), column=0, columnspan=2, sticky="w", pady=8)
            ttk.Label(frame, text="Key values remain in memory only; configuration saves the key name.",
                      foreground="#666").grid(row=len(fields) + 1, column=0, columnspan=2, sticky="w")
        notebook.select(selected)
        buttons = ttk.Frame(self, padding=(12, 0, 12, 12))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=self.save).pack(side="right", padx=8)

    def save(self):
        try:
            profiles, keys = {}, {}
            for role, variables in self.variables.items():
                profiles[role] = {
                    "provider": variables["provider"].get().strip(),
                    "model": variables["model"].get().strip(),
                    "base_url": variables["base_url"].get().strip(),
                    "api_key_env": variables["api_key_env"].get().strip(),
                    "temperature": float(variables["temperature"].get()),
                    "native_web_search": variables["native_web_search"].get(),
                }
                keys[role] = variables["api_key"].get().strip()
                if not profiles[role]["model"] or not profiles[role]["api_key_env"]:
                    raise ValueError(f"{ROLE_LABELS[role]} requires a model and key environment name.")
            self.store.update_all(profiles, keys)
        except ValueError as exc:
            messagebox.showerror("API configurations", str(exc), parent=self)
            return
        if self.on_saved:
            self.on_saved()
        self.destroy()


class AgentPanel(ttk.LabelFrame):
    def __init__(self, parent, role: str, configure_api: Callable[[str], None]):
        super().__init__(parent, text=ROLE_LABELS[role], padding=7)
        self.role, self.attachments = role, []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="we")
        self.api_var = tk.StringVar(value="API: not configured")
        ttk.Label(header, textvariable=self.api_var).pack(side="left")
        ttk.Button(header, text="API…", command=lambda: configure_api(role)).pack(side="right")
        tools = ttk.Frame(self)
        tools.grid(row=1, column=0, sticky="we", pady=(4, 0))
        ttk.Button(tools, text="Add Text", command=self.add_text).pack(side="left")
        ttk.Button(tools, text="Add File / Image", command=self.add_file).pack(side="left", padx=4)
        ttk.Button(tools, text="Copy Output", command=self.copy_output).pack(side="right")
        self.guidance = scrolledtext.ScrolledText(self, wrap="word", height=3, undo=True)
        self.guidance.grid(row=2, column=0, sticky="nsew", pady=4)
        self.output = scrolledtext.ScrolledText(self, wrap="word", height=7, undo=True)
        self.output.grid(row=3, column=0, sticky="nsew")
        bottom = ttk.Frame(self)
        bottom.grid(row=4, column=0, sticky="we", pady=(4, 0))
        self.attachment_var = tk.StringVar(value="No attachments")
        ttk.Label(bottom, textvariable=self.attachment_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bottom, text="Remove All", command=self.clear_attachments).pack(side="right")

    def add_text(self):
        value = simpledialog.askstring(ROLE_LABELS[self.role], "Add guidance or context:", parent=self)
        if value:
            if self.guidance.get("1.0", "end").strip():
                self.guidance.insert("end", "\n")
            self.guidance.insert("end", value)

    def add_file(self):
        names = filedialog.askopenfilenames(parent=self, title=f"Attach to {ROLE_LABELS[self.role]}",
            filetypes=[("Supported", "*.txt *.md *.json *.csv *.py *.png *.jpg *.jpeg *.gif *.webp"),
                       ("All files", "*.*")])
        existing = {str(path) for path in self.attachments}
        for name in names:
            if name not in existing:
                self.attachments.append(Path(name)); existing.add(name)
        self._update_attachments()

    def clear_attachments(self):
        self.attachments.clear(); self._update_attachments()

    def _update_attachments(self):
        names = ", ".join(path.name for path in self.attachments[:3])
        self.attachment_var.set("No attachments" if not names else f"Attached: {names}")

    def context(self):
        return {"guidance": self.guidance.get("1.0", "end").strip(),
                "attachments": list(self.attachments)}

    def set_output(self, value: Any):
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        self.output.delete("1.0", "end"); self.output.insert("end", text)

    def copy_output(self):
        text = self.output.get("1.0", "end").strip()
        if text:
            self.clipboard_clear(); self.clipboard_append(text); self.update_idletasks()

    def set_api_label(self, profile):
        self.api_var.set(f"API: {profile.get('model', 'not configured')}")


class ModelStageWindow(tk.Toplevel):
    """Normal chat on the left; invisible-to-chat fact checking on the right."""

    def __init__(self, parent, on_closed: Callable[[], None] | None = None):
        super().__init__(parent)
        self.title("ValenceSphere v11 — Chat Fact-Check Workspace")
        self.geometry("1700x980")
        self.minsize(1280, 760)
        self.on_closed = on_closed
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.profile_store = APIProfileStore()
        self.chat_messages: list[dict[str, str]] = []
        self.session: list[dict[str, Any]] = []
        self.question_history: list[str] = []
        self.history_index = 0
        self.current_folder: Path | None = None
        self.actual_tokens = 0
        self.chat_busy = False
        self.audit_jobs = 0
        self.cards: dict[str, dict[str, Any]] = {}
        self.active_concept: str | None = None
        self.latest_adjudication: dict[str, Any] | None = None
        self.audit_generation = 0
        self.turn_best_support: float | None = None
        self.turn_best_priority = 99
        self.graph_scale = 1.0
        self.engine = FactCheckEngine(self.client_for)
        self._styles(); self._menu(); self._interface(); self._context_menu()
        self._update_api_labels(); self._update_status(); self.render_graph()
        self.add_chat_card("System", "Normal chat is ready. ValenceSphere scans factual assertions in the background.",
                           "system", record=False)

    def client_for(self, role: str):
        return AgentAPIClient(self.profile_store.get(role))

    def _styles(self):
        style = ttk.Style(self)
        style.configure("Dark.TFrame", background="#0f172a")
        style.configure("Dark.TLabel", background="#0f172a", foreground="#e2e8f0",
                        font=("Segoe UI", 11, "bold"))

    def _menu(self):
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New Folder…", command=self.new_folder)
        file_menu.add_command(label="Load Folder…", command=self.load_folder)
        file_menu.add_command(label="Delete Folder…", command=self.delete_folder)
        file_menu.add_separator(); file_menu.add_command(label="Save Chat Markdown", command=self.save_chat)
        file_menu.add_separator(); file_menu.add_command(label="API Configurations…", command=self.configure_api)
        file_menu.add_separator(); file_menu.add_command(label="Close", command=self.close)
        menu.add_cascade(label="File", menu=file_menu); self.config(menu=menu)

    def _interface(self):
        toolbar = ttk.Frame(self, padding=(10, 7)); toolbar.pack(fill="x")
        ttk.Label(toolbar, text="STAGE 2 — NORMAL CHAT + BACKGROUND FACT CHECK",
                  font=("Segoe UI", 11, "bold")).pack(side="left")
        self.folder_var = tk.StringVar(value="Folder: not selected")
        ttk.Label(toolbar, textvariable=self.folder_var).pack(side="left", padx=14)
        ttk.Button(toolbar, text="Save Chat .md", command=self.save_chat).pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")
        main = ttk.PanedWindow(self, orient="horizontal"); main.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        left, right = ttk.Frame(main, padding=5), ttk.Frame(main, padding=5)
        main.add(left, weight=5); main.add(right, weight=7)
        self._chat(left); self._triad(right)

    def _chat(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent, style="Dark.TFrame", padding=8); header.grid(row=0, column=0, sticky="we")
        ttk.Label(header, text="CHAT", style="Dark.TLabel").pack(side="left")
        self.token_var = tk.StringVar(); ttk.Label(header, textvariable=self.token_var,
                                                   style="Dark.TLabel").pack(side="right")
        area = tk.Frame(parent, bg="#0f172a"); area.grid(row=1, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1); area.rowconfigure(0, weight=1)
        self.chat_canvas = tk.Canvas(area, bg="#0f172a", highlightthickness=0); self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(area, orient="vertical", command=self.chat_canvas.yview); scroll.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.configure(yscrollcommand=scroll.set)
        self.chat_inner = tk.Frame(self.chat_canvas, bg="#0f172a")
        self.chat_window_id = self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor="nw")
        self.chat_inner.bind("<Configure>", lambda _event: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.bind("<Configure>", lambda event: self.chat_canvas.itemconfigure(self.chat_window_id, width=event.width))
        compose = ttk.Frame(parent, padding=(0, 7, 0, 0)); compose.grid(row=2, column=0, sticky="we"); compose.columnconfigure(1, weight=1)
        ttk.Button(compose, text="↑ Previous Question", command=self.previous_question).grid(row=0, column=0, sticky="w")
        self.chat_input = scrolledtext.ScrolledText(compose, wrap="word", height=7, undo=True)
        self.chat_input.grid(row=1, column=0, columnspan=2, sticky="we", pady=5)
        self.chat_input.bind("<KeyRelease>", lambda _event: self._tokens())
        self.chat_input.bind("<Control-Return>", lambda _event: self.send_chat())
        self.send_button = ttk.Button(compose, text="Send", command=self.send_chat); self.send_button.grid(row=2, column=1, sticky="e")
        ttk.Label(compose, text="Ctrl+Enter sends • highlights appear after background verification").grid(row=2, column=0, sticky="w")

    def _triad(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(1, weight=1)
        top = ttk.Frame(parent, style="Dark.TFrame", padding=8); top.grid(row=0, column=0, sticky="we")
        ttk.Label(top, text="VALENCESPHERE SCANNER", style="Dark.TLabel").pack(side="left")
        self.active_var = tk.StringVar(value="Active concept: none")
        ttk.Label(top, textvariable=self.active_var, style="Dark.TLabel").pack(side="right")
        vertical = ttk.PanedWindow(parent, orient="vertical"); vertical.grid(row=1, column=0, sticky="nsew")
        upper = ttk.PanedWindow(vertical, orient="horizontal")
        self.questioner_panel = AgentPanel(upper, "questioner", self.configure_api)
        self.answerer_panel = AgentPanel(upper, "answerer", self.configure_api)
        upper.add(self.questioner_panel, weight=1); upper.add(self.answerer_panel, weight=1); vertical.add(upper, weight=1)
        lower = ttk.PanedWindow(vertical, orient="horizontal")
        self.adjudicator_panel = AgentPanel(lower, "adjudicator", self.configure_api)
        graph = ttk.LabelFrame(lower, text="Assertion Knowledge Graph", padding=5)
        graph.columnconfigure(0, weight=1); graph.rowconfigure(1, weight=1)
        tools = ttk.Frame(graph); tools.grid(row=0, column=0, sticky="we")
        ttk.Button(tools, text="+", width=3, command=lambda: self.zoom_graph(1.2)).pack(side="left")
        ttk.Button(tools, text="−", width=3, command=lambda: self.zoom_graph(1 / 1.2)).pack(side="left")
        ttk.Button(tools, text="Reset", command=self.reset_graph).pack(side="left", padx=4)
        self.graph_note_var = tk.StringVar(value="Click a node for its audit note.")
        ttk.Label(tools, textvariable=self.graph_note_var).pack(side="left", padx=6)
        self.graph_canvas = tk.Canvas(graph, bg="#111827", highlightthickness=0); self.graph_canvas.grid(row=1, column=0, sticky="nsew")
        self.graph_canvas.bind("<ButtonPress-1>", lambda event: self.graph_canvas.scan_mark(event.x, event.y))
        self.graph_canvas.bind("<B1-Motion>", lambda event: self.graph_canvas.scan_dragto(event.x, event.y, gain=1))
        self.graph_canvas.bind("<MouseWheel>", lambda event: self.zoom_graph(1.12 if event.delta > 0 else 1 / 1.12))
        lower.add(self.adjudicator_panel, weight=1); lower.add(graph, weight=1); vertical.add(lower, weight=1)

    def _context_menu(self):
        self.context = tk.Menu(self, tearoff=False); self._context_widget = None
        for label, action in (("Copy", "copy"), ("Paste", "paste"), ("Delete", "delete"),
                              ("Undo", "undo"), ("Redo", "redo")):
            self.context.add_command(label=label, command=lambda value=action: self._context_action(value))
        self.bind("<Button-3>", self._show_context)

    def _show_context(self, event):
        self._context_widget = event.widget; self.context.tk_popup(event.x_root, event.y_root)

    def _context_action(self, action):
        widget = self._context_widget
        if not widget: return
        events = {"copy": "<<Copy>>", "paste": "<<Paste>>", "undo": "<<Undo>>", "redo": "<<Redo>>"}
        try:
            if action in events: widget.event_generate(events[action])
            elif action == "delete":
                if isinstance(widget, tk.Text): widget.delete("sel.first", "sel.last")
                elif isinstance(widget, (tk.Entry, ttk.Entry)): widget.delete("sel.first", "sel.last")
        except tk.TclError: pass

    def add_chat_card(
            self,
            title,
            text,
            kind,
            message_id=None,
            record=True,
    ):
        message_id = message_id or f"msg_{uuid.uuid4().hex[:10]}"

        palette = {
            "user": ("#2563eb", "#eff6ff"),
            "assistant": ("#0f766e", "#ecfeff"),
            "system": ("#64748b", "#f8fafc"),
            "error": ("#b91c1c", "#fef2f2"),
        }

        accent, background = palette.get(kind, palette["assistant"])

        card = tk.Frame(
            self.chat_inner,
            bg=background,
            highlightbackground=accent,
            highlightthickness=2,
            padx=9,
            pady=7,
        )
        card.pack(fill="x", expand=True, padx=9, pady=6)

        head = tk.Frame(card, bg=background)
        head.pack(fill="x")

        title_label = tk.Label(
            head,
            text=title,
            bg=background,
            fg=accent,
            font=("Segoe UI", 10, "bold"),
        )
        title_label.pack(side="left")

        audit_label = tk.Label(
            head,
            text="",
            bg=background,
            fg="#475569",
            font=("Segoe UI", 9, "bold"),
        )
        audit_label.pack(side="left", padx=8)

        ttk.Button(
            head,
            text="Copy",
            command=lambda value=text: self.copy_text(value),
        ).pack(side="right")

        body = tk.Label(
            card,
            text=text,
            bg=background,
            fg="#111827",
            justify="left",
            anchor="nw",
            wraplength=320,
            font=("Segoe UI", 10),
        )
        body.pack(fill="x", expand=True, pady=(5, 0))

        def resize_message(event):
            available_width = max(220, event.width - 26)
            body.configure(wraplength=available_width)

        card.bind("<Configure>", resize_message)

        self.cards[message_id] = {
            "card": card,
            "head": head,
            "title": title_label,
            "audit": audit_label,
            "body": body,
            "background": background,
        }

        if record:
            self.session.append({
                "id": message_id,
                "title": title,
                "text": text,
                "kind": kind,
                "timestamp": now_iso(),
            })

        self.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))
        self._tokens()

        return message_id

    def highlight_card(self, message_id: str, outcomes: list[FactCheckOutcome]):
        card = self.cards.get(message_id)
        if not card or not outcomes: return
        scores = [float(item.adjudication.get("final", {}).get("assertion_support", 50.0))
                  for item in outcomes if item.adjudication]
        if not scores: return
        category = confidence_category(min(scores)); colour = category["colour"]
        card["card"].configure(highlightbackground=colour, highlightthickness=4)
        card["audit"].configure(text=f"● {category['label']} ({min(scores):.1f}%)", fg=colour)

    def send_chat(self):
        if self.chat_busy: return
        prompt = self.chat_input.get("1.0", "end").strip()
        if not prompt: return
        self.question_history.append(prompt); self.history_index = len(self.question_history)
        self.chat_input.delete("1.0", "end")
        user_id = self.add_chat_card("You", prompt, "user")
        self.chat_messages.append({"role": "user", "content": prompt})
        history = list(self.chat_messages[-20:])
        contexts = self._agent_contexts()
        self.chat_busy = True; self.send_button.config(state="disabled"); self._update_status()
        threading.Thread(target=self._chat_worker, args=(history, user_id, prompt, contexts), daemon=True).start()

    def _chat_worker(self, history, user_id, user_text, contexts):
        try:
            response = self.client_for("chat").chat(history)
            self.after(
                0,
                lambda: self._finish_chat(
                    user_id,
                    user_text,
                    response,
                    contexts,
                    None,
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            self.after(
                0,
                lambda error=error_message: self._finish_chat(
                    user_id,
                    user_text,
                    None,
                    contexts,
                    error,
                ),
            )

    def _finish_chat(self, user_id, user_text, response, contexts, error):
        self.chat_busy = False
        self.send_button.config(state="normal")

        if error:
            self.add_chat_card("Chat error", error, "error")
            self._update_status()
            return

        assistant_id = self.add_chat_card(
            "Assistant",
            response.text,
            "assistant",
        )

        self.chat_messages.append({
            "role": "assistant",
            "content": response.text,
        })

        self.actual_tokens += response.usage.get("total_tokens", 0)

        # Begin a new audit generation for this user/assistant exchange.
        # Both messages are scanned, but the least-supported assertion remains
        # active in the triad and Knowledge Graph.
        self.audit_generation += 1
        generation = self.audit_generation

        self.turn_best_support = None
        self.turn_best_priority = 99

        self._start_audit(
            user_id,
            user_text,
            "user",
            contexts,
            generation,
        )

        self._start_audit(
            assistant_id,
            response.text,
            "assistant",
            contexts,
            generation,
        )

        self._tokens()
        self._update_status()

    def _start_audit(
            self,
            message_id,
            text,
            speaker,
            contexts,
            generation,
    ):
        self.audit_jobs += 1
        self._update_status()

        threading.Thread(
            target=self._audit_worker,
            args=(
                message_id,
                text,
                speaker,
                contexts,
                generation,
            ),
            daemon=True,
        ).start()

    def _audit_worker(
            self,
            message_id,
            text,
            speaker,
            contexts,
            generation,
    ):
        try:
            outcomes = self.engine.scan_message(
                text,
                speaker,
                message_id,
                contexts,
            )

            self.after(
                0,
                lambda: self._finish_audit(
                    message_id,
                    outcomes,
                    None,
                    generation,
                ),
            )

        except Exception as exc:
            error_message = str(exc)

            self.after(
                0,
                lambda error=error_message: self._finish_audit(
                    message_id,
                    [],
                    error,
                    generation,
                ),
            )

    def _finish_audit(
            self,
            message_id,
            outcomes,
            error,
            generation,
    ):
        self.audit_jobs = max(0, self.audit_jobs - 1)

        if error:
            card = self.cards.get(message_id)

            if card:
                card["audit"].configure(
                    text=f"Audit unavailable: {error}",
                    fg="#b91c1c",
                )

            self._update_status()
            return

        if not outcomes:
            self._update_status()
            return

        # Colour the individual chat card according to its own audit.
        self.highlight_card(message_id, outcomes)

        # An older audit may finish after a newer chat exchange has started.
        # It can colour its original card, but it must not replace the current KG.
        if generation != self.audit_generation:
            self._update_status()
            return

        def outcome_support(outcome):
            if not outcome.adjudication:
                return 50.0

            return float(
                outcome.adjudication
                .get("final", {})
                .get("assertion_support", 50.0)
            )

        def outcome_priority(outcome):
            questioner = outcome.answerer.get("questioner", {})
            message = questioner.get("message", {})
            speaker = str(message.get("speaker", "")).casefold()

            # When confidence scores tie, retain the user's assertion.
            return 0 if speaker == "user" else 1

        candidate = min(
            outcomes,
            key=lambda outcome: (
                outcome_support(outcome),
                outcome_priority(outcome),
            ),
        )

        candidate_support = outcome_support(candidate)
        candidate_priority = outcome_priority(candidate)

        is_better_candidate = (
                self.turn_best_support is None
                or candidate_support < self.turn_best_support
                or (
                        abs(candidate_support - self.turn_best_support) < 0.001
                        and candidate_priority < self.turn_best_priority
                )
        )

        if not is_better_candidate:
            self._update_status()
            return

        self.turn_best_support = candidate_support
        self.turn_best_priority = candidate_priority

        self.active_concept = candidate.concept
        self.active_var.set(f"Active concept: {candidate.concept}")

        self.questioner_panel.set_output(
            candidate.answerer["questioner"]
        )

        self.answerer_panel.set_output(
            candidate.answerer["answerer"]
        )

        if candidate.adjudication:
            self.latest_adjudication = candidate.adjudication

            self.adjudicator_panel.set_output({
                "cached": candidate.cached,
                "assertion": candidate.adjudication.get("assertion", ""),
                "spectrum": (
                    candidate.adjudication
                    .get("verification", {})
                    .get("spectrum", {})
                ),
                "final": candidate.adjudication.get("final", {}),
            })

        self.render_graph()
        self._update_status()

    def _agent_contexts(self):
        return {"questioner": self.questioner_panel.context(),
                "answerer": self.answerer_panel.context(),
                "adjudicator": self.adjudicator_panel.context()}

    def render_graph(self):
        canvas = self.graph_canvas; canvas.delete("all")
        if self.active_concept and self.latest_adjudication:
            graph = build_assertion_graph(self.active_concept, self.latest_adjudication)
            positions = {"concept": (140, 220), "asserted": (420, 110), "corrected": (420, 330)}
            for edge in graph["edges"]:
                x1, y1 = positions[edge["source"]]; x2, y2 = positions[edge["target"]]
                canvas.create_line(x1, y1, x2, y2, fill="#94a3b8", arrow="last", width=2)
                canvas.create_text((x1+x2)/2, (y1+y2)/2, text=edge["label"], fill="#e2e8f0")
            for index, node in enumerate(graph["nodes"]):
                x, y = positions[node["id"]]; tag = f"audit_node_{index}"
                canvas.create_oval(x-78, y-42, x+78, y+42, fill=node["colour"], outline="#f8fafc", width=3, tags=tag)
                canvas.create_text(x, y, text=node["label"], width=135, fill="#111827",
                                   font=("Segoe UI", 9, "bold"), tags=tag)
                canvas.tag_bind(tag, "<Button-1>", lambda _e, note=node["note"]: self.graph_note_var.set(note))
        else:
            records = discover_concepts()
            for index, record in enumerate(records):
                x, y = 120 + (index % 3) * 210, 100 + (index // 3) * 140
                canvas.create_oval(x-60, y-34, x+60, y+34, fill="#1e293b", outline="#64748b", width=2)
                canvas.create_text(x, y, text=record.name, fill="#e2e8f0", width=105)
        canvas.scale("all", 0, 0, self.graph_scale, self.graph_scale)
        canvas.configure(scrollregion=canvas.bbox("all") or (0, 0, 900, 600))

    def zoom_graph(self, factor):
        self.graph_scale = max(.4, min(3.0, self.graph_scale * factor)); self.render_graph()

    def reset_graph(self):
        self.graph_scale = 1.0; self.render_graph(); self.graph_canvas.xview_moveto(0); self.graph_canvas.yview_moveto(0)

    def previous_question(self):
        if not self.question_history: return
        self.history_index = max(0, self.history_index - 1)
        self.chat_input.delete("1.0", "end"); self.chat_input.insert("end", self.question_history[self.history_index]); self._tokens()

    def _tokens(self):
        current = self.chat_input.get("1.0", "end").strip() if hasattr(self, "chat_input") else ""
        estimated = sum(max(1, len(item.get("text", "")) // 4) for item in self.session) + len(current)//4
        self.token_var.set(f"≈ {estimated:,} chat tokens • {self.actual_tokens:,} API tokens")

    def _update_status(self):
        concepts = len(discover_concepts())
        activity = "chatting" if self.chat_busy else f"{self.audit_jobs} audit(s) running" if self.audit_jobs else "ready"
        self.status_var.set(f"{concepts} concepts • {activity}")

    def refresh_knowledge(self):
        """Refresh the live Stage 1 repository view when the pop-out is reopened."""
        self.render_graph()
        self._update_status()

    def configure_api(self, role="chat"):
        APIConfigurationDialog(self, self.profile_store, role, self._update_api_labels)

    def _update_api_labels(self):
        self.questioner_panel.set_api_label(self.profile_store.get("questioner"))
        self.answerer_panel.set_api_label(self.profile_store.get("answerer"))
        sources = ", ".join(self.profile_store.get(role)["model"] for role in ("source_1", "source_2", "source_3"))
        adjudicator = self.profile_store.get("adjudicator")
        adjudicator["model"] = f"{adjudicator['model']} • sources: {sources}"
        self.adjudicator_panel.set_api_label(adjudicator)

    def copy_text(self, text):
        self.clipboard_clear(); self.clipboard_append(text); self.update_idletasks()

    def new_folder(self):
        name = simpledialog.askstring("New chat folder", "Folder name:", parent=self)
        if not name: return
        folder = WORKSPACE_ROOT / safe_filename(name)
        try: folder.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if not messagebox.askyesno("Folder exists", "Use the existing folder?", parent=self): return
        self.current_folder = folder; self.folder_var.set(f"Folder: {folder.name}")

    def load_folder(self):
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askdirectory(parent=self, initialdir=WORKSPACE_ROOT)
        if not selected: return
        folder = Path(selected); payload = load_json(folder / "session.json", {})
        self.current_folder = folder; self.folder_var.set(f"Folder: {folder.name}")
        if isinstance(payload, dict):
            self.clear_chat(); self.session = []; self.chat_messages = payload.get("chat_messages", [])
            for item in payload.get("session", []):
                self.add_chat_card(item.get("title", "Loaded"), item.get("text", ""), item.get("kind", "system"),
                                   item.get("id"), record=True)
            self.question_history = payload.get("question_history", []); self.history_index = len(self.question_history)

    def delete_folder(self):
        if not self.current_folder: return
        try: self.current_folder.resolve().relative_to(WORKSPACE_ROOT.resolve())
        except ValueError:
            messagebox.showerror("Delete folder", "Only Stage 2 chat folders can be deleted.", parent=self); return
        if messagebox.askyesno("Delete folder", f"Permanently delete '{self.current_folder.name}'?", parent=self):
            shutil.rmtree(self.current_folder); self.current_folder = None; self.folder_var.set("Folder: not selected")

    def save_chat(self):
        if not self.current_folder: self.new_folder()
        if not self.current_folder: return
        lines = ["# ValenceSphere Chat", "", f"Saved: {now_iso()}", ""]
        for item in self.session: lines.extend((f"## {item['title']}", "", item["text"], ""))
        atomic_write_text(self.current_folder / "chat.md", "\n".join(lines).rstrip()+"\n")
        atomic_write_json(self.current_folder / "session.json", {"version": 1, "saved_at": now_iso(),
            "chat_messages": self.chat_messages, "question_history": self.question_history, "session": self.session})
        self.folder_var.set(f"Folder: {self.current_folder.name} • saved")

    def clear_chat(self):
        for child in self.chat_inner.winfo_children(): child.destroy()
        self.cards.clear()

    def close(self):
        if (self.chat_busy or self.audit_jobs) and not messagebox.askyesno(
                "Close", "Chat or background verification is still running. Close?", parent=self): return
        self.destroy()
        if self.on_closed: self.on_closed()
