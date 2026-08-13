"""Tkinter interface for the streamlined ValenceSphere architecture."""
from __future__ import annotations

import json
import shutil
import re
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from concept_repository import (
    concept_path,
    discover_concepts,
    find_concept,
    load_spawn_queue,
    save_question_constituents,
    save_spawn_queue,
)
from learning_controller import LearningController
from llm_services import ConstituentAnalysisService
from model_stage import ModelStageWindow
from spawn_service import SpawnService
from template_schema import ensure_template_file, now_iso
from utils import normalize_concept_name
from constants import LOG_DIR

class ValenceSphereGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ValenceSphere v11 — Stage 1: Concept Formation")
        self.geometry("1400x950")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.current_concept: str | None = None
        self.current_path: Path | None = None
        self.controller: LearningController | None = None
        self.running_controller: LearningController | None = None
        self.model_stage_window: ModelStageWindow | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pending_question: dict | None = None
        self.analyzer_questions: list[dict] = []
        self.status_var = tk.StringVar(value="Ready.")
        self.new_concept_var = tk.StringVar()
        self.select_all_var = tk.BooleanVar(value=False)
        self.spawn_service = SpawnService()
        self._build()
        self.refresh_library()

    def _build(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self._build_library()
        self._build_concept()
        self._build_qa()
        self._build_analyzer()
        self._build_spawns()
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(fill="x", side="bottom")
        ttk.Label(bar, textvariable=self.status_var).pack(side="left")
        ttk.Button(bar, text="Open Stage 2 — Model Workspace",
                   command=self.open_model_stage).pack(side="right")

    def _build_library(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Library")

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        row = ttk.Frame(frame)
        row.grid(
            row=0,
            column=0,
            sticky="we",
            pady=(0, 6),
        )

        ttk.Entry(
            row,
            textvariable=self.new_concept_var,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            row,
            text="Add",
            command=self.add_concept,
        ).pack(
            side="left",
            padx=5,
        )

        ttk.Button(
            row,
            text="Delete",
            command=self.delete_concepts,
        ).pack(side="left")

        controls = ttk.Frame(frame)
        controls.grid(
            row=1,
            column=0,
            sticky="we",
            pady=(0, 6),
        )

        ttk.Checkbutton(
            controls,
            text="Select All",
            variable=self.select_all_var,
            command=self.toggle_select_all,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Learn Selected",
            command=self.learn_selected,
        ).pack(
            side="left",
            padx=6,
        )

        ttk.Button(
            controls,
            text="Stop",
            command=self.stop_learning,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Refresh",
            command=self.refresh_library,
        ).pack(
            side="left",
            padx=6,
        )

        self.progress = ttk.Progressbar(
            controls,
            length=220,
            mode="determinate",
        )
        self.progress.pack(
            side="left",
            padx=8,
        )

        # Persistent visual feedback for the learning process.
        self.learning_status_var = tk.StringVar(
            value="Learning idle."
        )

        self.learning_status = ttk.Label(
            controls,
            textvariable=self.learning_status_var,
            foreground="#64748b",
            font=("Segoe UI", 9, "bold"),
        )
        self.learning_status.pack(
            side="left",
            padx=(8, 0),
        )

        self.library_tree = ttk.Treeview(
            frame,
            columns=("path", "concept"),
            show="tree",
            selectmode="extended",
        )
        self.library_tree.grid(
            row=2,
            column=0,
            sticky="nsew",
        )

        self.library_tree.bind(
            "<Double-1>",
            self.open_selected_concept,
        )
        self.library_tree.bind(
            "<Return>",
            self.open_selected_concept,
        )

        scroll = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.library_tree.yview,
        )
        scroll.grid(
            row=2,
            column=1,
            sticky="ns",
        )

        self.library_tree.configure(
            yscrollcommand=scroll.set,
        )

        self.library_tree.tag_configure(
            "incomplete",
            foreground="#b45309",
        )

        self.library_tree.tag_configure(
            "new",
            foreground="#2563eb",
            font=("Segoe UI", 9, "bold"),
        )

        self.library_tree.tag_configure(
            "learned",
            foreground="#166534",
            background="#dcfce7",
            font=("Segoe UI", 9, "bold"),
        )

    def _build_concept(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Concept")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=3)
        frame.rowconfigure(2, weight=2)
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="we", pady=(0, 6))
        self.concept_label = ttk.Label(top, text="No concept selected", font=("Segoe UI", 10, "bold"))
        self.concept_label.pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_concept).pack(side="left", padx=6)
        panes = ttk.PanedWindow(frame, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew")
        tree_box = ttk.Frame(panes)
        tree_box.rowconfigure(0, weight=1)
        tree_box.columnconfigure(0, weight=1)
        self.template_tree = ttk.Treeview(tree_box, show="tree")
        self.template_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(tree_box, orient="vertical", command=self.template_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.template_tree.configure(yscrollcommand=tree_scroll.set)
        self.template_tree.tag_configure("answered", foreground="#15803d")
        panes.add(tree_box, weight=2)
        log_box = ttk.LabelFrame(panes, text="Activity Log")
        self.log_text = scrolledtext.ScrolledText(log_box, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        panes.add(log_box, weight=2)
        self.template_text = scrolledtext.ScrolledText(frame, wrap="none")
        self.template_text.grid(row=2, column=0, sticky="nsew", pady=(6, 0))

    def _build_qa(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Q & A")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)
        self.qa_phase_var = tk.StringVar(value="Phase: —")
        ttk.Label(frame, textvariable=self.qa_phase_var, font=("Segoe UI", 9, "italic")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(frame, text="Question", font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w")
        self.qa_question = scrolledtext.ScrolledText(frame, wrap="word", height=4)
        self.qa_question.grid(row=2, column=0, sticky="nsew")
        self.qa_context = ttk.Label(frame, text="", foreground="#666", wraplength=1200, justify="left")
        self.qa_context.grid(row=3, column=0, sticky="we", pady=(4, 8))
        ttk.Label(frame, text="Accepted answer", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w")
        self.qa_answer = scrolledtext.ScrolledText(frame, wrap="word", height=10)
        self.qa_answer.grid(row=5, column=0, sticky="nsew")
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, sticky="we", pady=(8, 0))
        ttk.Button(buttons, text="Prepare Next", command=self.prepare_question).pack(side="left")
        self.qa_send = ttk.Button(buttons, text="Send to API", command=self.send_question)
        self.qa_send.pack(side="left", padx=6)
        ttk.Button(buttons, text="Auto-Learn", command=self.learn_current).pack(side="left")
        ttk.Button(buttons, text="Stop", command=self.stop_learning).pack(side="left", padx=6)

    def _build_analyzer(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="Analyzer")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(1, weight=1)

        ttk.Label(
            frame, text="Concept questions", font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame, text="LLM constituent analysis", font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=1, sticky="w")

        left = ttk.Frame(frame)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.analyzer_list = tk.Listbox(left, exportselection=False)
        self.analyzer_list.grid(row=0, column=0, sticky="nsew")
        self.analyzer_list.bind("<<ListboxSelect>>", self.select_analysis_question)
        ttk.Button(left, text="Refresh", command=self.refresh_analyzer).grid(
            row=1, column=0, sticky="we", pady=(6, 0)
        )

        right = ttk.Frame(frame)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)
        right.rowconfigure(7, weight=1)

        ttk.Label(right, text="Question").grid(row=0, column=0, sticky="w")
        self.analyzer_question = scrolledtext.ScrolledText(right, wrap="word", height=3)
        self.analyzer_question.grid(row=1, column=0, sticky="nsew")

        ttk.Label(right, text="Accepted answer (semantic context)").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.analyzer_answer = scrolledtext.ScrolledText(right, wrap="word", height=5)
        self.analyzer_answer.grid(row=3, column=0, sticky="nsew")

        ttk.Label(right, text="Meaning-bearing constituents").grid(
            row=4, column=0, sticky="w", pady=(6, 0)
        )
        self.analyzer_constituents = scrolledtext.ScrolledText(right, wrap="word", height=8)
        self.analyzer_constituents.grid(row=5, column=0, sticky="nsew")

        ttk.Label(right, text="Saved constituent JSON").grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )
        self.analyzer_json = scrolledtext.ScrolledText(right, wrap="none", height=8)
        self.analyzer_json.grid(row=7, column=0, sticky="nsew")

        buttons = ttk.Frame(right)
        buttons.grid(row=8, column=0, sticky="we", pady=(8, 0))
        self.analyze_button = ttk.Button(buttons, text="Analyze Selected with API", command=self.analyze_selected)
        self.analyze_button.pack(side="left")
        ttk.Button(buttons, text="Analyze All Answered", command=self.analyze_all).pack(side="left", padx=6)
        self.analyzer_status = ttk.Label(buttons, text="", foreground="#00695c")
        self.analyzer_status.pack(side="left", padx=8)

    def _build_spawns(self):
        frame = ttk.Frame(
            self.notebook,
            padding=10,
        )

        self.notebook.add(
            frame,
            text="Spawns",
        )

        frame.columnconfigure(
            0,
            weight=1,
        )
        frame.rowconfigure(
            1,
            weight=1,
        )

        ttk.Label(
            frame,
            text="Spawn candidates",
            font=("Segoe UI", 11, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        self.spawn_list = tk.Listbox(
            frame,
            selectmode="extended",
            exportselection=False,
        )
        self.spawn_list.grid(
            row=1,
            column=0,
            sticky="nsew",
            pady=6,
        )

        # Double-clicking a candidate also opens the editor.
        self.spawn_list.bind(
            "<Double-1>",
            self.edit_selected_spawn,
        )

        controls = ttk.Frame(frame)
        controls.grid(
            row=2,
            column=0,
            sticky="we",
        )

        ttk.Button(
            controls,
            text="Refresh",
            command=self.refresh_spawns,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Edit Selected",
            command=self.edit_selected_spawn,
        ).pack(
            side="left",
            padx=6,
        )

        ttk.Button(
            controls,
            text="Auto Spawn",
            command=self.auto_spawn,
        ).pack(side="left")

        ttk.Button(
            controls,
            text="Remove Selected",
            command=self.remove_spawns,
        ).pack(
            side="left",
            padx=6,
        )

        self.spawn_status = ttk.Label(
            controls,
            text="",
        )
        self.spawn_status.pack(
            side="left",
            padx=8,
        )

    # Library and concept -------------------------------------------------
    def refresh_library(self, highlight=None):
        self.library_tree.delete(
            *self.library_tree.get_children()
        )

        records = discover_concepts()

        by_name = {
            record.name.casefold(): record
            for record in records
        }

        sequence = iter(
            range(1, len(records) + 1)
        )

        highlighted = {
            value.casefold()
            for value in (highlight or [])
        }

        visited = set()

        def add_children(
                parent_name,
                parent_item="",
        ):
            for record in sorted(
                    records,
                    key=lambda item: item.name.casefold(),
            ):
                if (
                        str(record.parent or "").casefold()
                        == parent_name.casefold()
                ):
                    add_record(
                        record,
                        parent_item,
                    )

        def add_record(
                record,
                parent_item="",
        ):
            key = record.name.casefold()

            if key in visited:
                return

            visited.add(key)

            questions = (
                record.data
                .get("discovery", {})
                .get("questions", [])
            )

            has_questions = bool(questions)

            learned = (
                    has_questions
                    and all(
                question.get("status") == "answered"
                for question in questions
            )
            )

            incomplete = (
                    not learned
                    and any(
                question.get("status") != "answered"
                for question in questions
            )
            )

            if key in highlighted:
                tags = ("new",)
            elif learned:
                tags = ("learned",)
            elif incomplete:
                tags = ("incomplete",)
            else:
                tags = ()

            item = self.library_tree.insert(
                parent_item,
                "end",
                text=f"{next(sequence)}. {record.name}",
                values=(
                    str(record.path),
                    record.name,
                ),
                tags=tags,
            )

            add_children(
                record.name,
                item,
            )

        for record in sorted(
                records,
                key=lambda item: item.name.casefold(),
        ):
            parent_key = str(
                record.parent or ""
            ).casefold()

            if not parent_key or parent_key not in by_name:
                add_record(record)

        # Invalid or circular parent information must not hide a concept.
        for record in sorted(
                records,
                key=lambda item: item.name.casefold(),
        ):
            if record.name.casefold() not in visited:
                add_record(record)

        learned_count = sum(
            bool(
                record.data
                .get("discovery", {})
                .get("questions", [])
            )
            and all(
                question.get("status") == "answered"
                for question in (
                    record.data
                    .get("discovery", {})
                    .get("questions", [])
                )
            )
            for record in records
        )

        self.status_var.set(
            f"Library refreshed "
            f"({len(records)} concepts, "
            f"{learned_count} learned)."
        )

    def all_tree_items(self, parent=""):
        result = []
        for item in self.library_tree.get_children(parent):
            result.append(item)
            result.extend(self.all_tree_items(item))
        return result

    def toggle_select_all(self):
        items = self.all_tree_items()
        self.library_tree.selection_set(items) if self.select_all_var.get() else self.library_tree.selection_remove(items)

    def selected_concepts(self):
        return [self.library_tree.item(item, "values")[1] for item in self.library_tree.selection()]

    def add_concept(self):
        name = self.new_concept_var.get().strip()
        if not name:
            self.status_var.set("Enter a concept name.")
            return
        path = concept_path(name)
        if path.exists():
            self.status_var.set("Concept already exists.")
            return
        ensure_template_file(str(path), name)
        self.new_concept_var.set("")
        self.refresh_library([name])

    def delete_concepts(self):
        names = self.selected_concepts()
        if not names or not messagebox.askyesno(
                "Delete concepts",
                f"Permanently delete {len(names)} selected concept director{'y' if len(names) == 1 else 'ies'} "
                "and all of their audit records?"):
            return
        for name in names:
            record = find_concept(name)
            if record:
                shutil.rmtree(record.directory)
        self.refresh_library()

    def open_selected_concept(self, _event=None):
        selection = self.library_tree.selection()
        if selection:
            self.open_concept(self.library_tree.item(selection[0], "values")[1])

    def open_concept(self, name: str):
        record = find_concept(name)
        if not record:
            return
        self.current_concept, self.current_path = record.name, record.path
        self.controller = LearningController(record.name, event_sink=self.engine_event)
        self.concept_label.config(text=f"Concept: {record.name}")
        self.refresh_concept()
        self.refresh_analyzer()
        self.notebook.select(self.notebook.tabs()[1])
        self.status_var.set(f"Loaded concept '{record.name}'.")

    def refresh_concept(self):
        if not self.current_concept:
            return
        record = find_concept(self.current_concept)
        if not record:
            return
        self.current_path = record.path
        self.template_tree.delete(*self.template_tree.get_children())
        root = self.template_tree.insert("", "end", text=record.name, open=True)
        phases = {}
        for question in record.data.get("discovery", {}).get("questions", []):
            phases.setdefault(question.get("phase", "discovery"), []).append(question)
        number = 1
        for phase, questions in phases.items():
            phase_item = self.template_tree.insert(root, "end", text=phase.title(), open=True)
            for question in questions:
                status = question.get("status", "unasked")
                self.template_tree.insert(phase_item, "end", text=f"{number:02d}. [{status}] {question.get('text', '')}",
                                          tags=("answered",) if status == "answered" else ())
                number += 1
        self.template_text.delete("1.0", "end")
        self.template_text.insert("end", json.dumps(record.data, indent=2, ensure_ascii=False))
        self.refresh_logs()

    def refresh_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")

        concept = str(
            self.current_concept or ""
        ).strip()

        if not concept:
            self.log_text.insert(
                "end",
                "Select a concept to view its adjudication activity.",
            )
            self.log_text.configure(state="disabled")
            return

        concept_key = concept.casefold()
        records = []

        if LOG_DIR.exists():
            for log_path in sorted(
                    LOG_DIR.glob("*.jsonl"),
                    reverse=True,
            ):
                # Conflict logs have a different record structure and should not
                # be mixed into the ordinary adjudication activity display.
                if log_path.name.startswith("conflicts_"):
                    continue

                try:
                    with log_path.open(
                            "r",
                            encoding="utf-8",
                    ) as handle:
                        for line_number, line in enumerate(
                                handle,
                                start=1,
                        ):
                            line = line.strip()

                            if not line:
                                continue

                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if not isinstance(record, dict):
                                continue

                            record_concept = str(
                                record.get("concept") or ""
                            ).strip()

                            if record_concept.casefold() != concept_key:
                                continue

                            record["_log_file"] = log_path.name
                            record["_line_number"] = line_number
                            records.append(record)

                except OSError:
                    continue

        records.sort(
            key=lambda record: str(
                record.get("timestamp") or ""
            ),
            reverse=True,
        )

        if not records:
            self.log_text.insert(
                "end",
                f"No adjudication activity has been recorded for "
                f"'{concept}'.",
            )
            self.log_text.configure(state="disabled")
            return

        for index, record in enumerate(records, start=1):
            timestamp = str(
                record.get("timestamp") or "Unknown time"
            )

            decision = str(
                record.get("decision") or "unknown"
            ).replace("_", " ").title()

            confidence = record.get("confidence")

            if isinstance(confidence, (int, float)):
                confidence_text = f"{confidence:.1f}%"
            else:
                confidence_text = "—"

            question = str(
                record.get("question") or ""
            ).strip()

            answer = str(
                record.get("answer") or ""
            ).strip()

            reasoning = str(
                record.get("reasoning") or ""
            ).strip()

            conflicts = record.get("related_conflicts") or []

            self.log_text.insert(
                "end",
                f"{index}. {timestamp}\n",
            )

            self.log_text.insert(
                "end",
                f"Decision: {decision}\n",
            )

            self.log_text.insert(
                "end",
                f"Confidence: {confidence_text}\n",
            )

            if question:
                self.log_text.insert(
                    "end",
                    f"Question: {question}\n",
                )

            if answer:
                self.log_text.insert(
                    "end",
                    f"Answer: {answer}\n",
                )

            if reasoning:
                self.log_text.insert(
                    "end",
                    f"Reasoning: {reasoning}\n",
                )

            if conflicts:
                self.log_text.insert(
                    "end",
                    f"Related conflicts: "
                    f"{json.dumps(conflicts, ensure_ascii=False)}\n",
                )

            self.log_text.insert(
                "end",
                f"Source log: {record['_log_file']}\n",
            )

            if index < len(records):
                self.log_text.insert(
                    "end",
                    "\n" + ("─" * 70) + "\n\n",
                )

        self.log_text.configure(state="disabled")
        self.log_text.see("1.0")

    # Learning ------------------------------------------------------------
    def engine_event(self, event, payload):
        self.after(0, lambda: self.status_var.set(payload.get("error") or event.title()))

    def prepare_question(self):
        if not self.controller:
            self.status_var.set("Select a concept first.")
            return
        question = self.controller.next_question()
        if not question:
            self.status_var.set("All discovery questions are answered.")
            return
        self.pending_question = question
        self.qa_question.delete("1.0", "end")
        self.qa_question.insert("end", question.get("text", ""))
        self.qa_answer.delete("1.0", "end")
        self.qa_phase_var.set(f"Phase: {question.get('phase', 'discovery')}")
        context = question.get("context", {}).get("text", "")
        self.qa_context.config(text=("Accepted definition used as background context: " + context) if context else "")
        self.status_var.set("Question prepared.")

    def send_question(self):
        if not self.controller:
            self.status_var.set("Select a concept first.")
            return
        text = self.qa_question.get("1.0", "end").strip()
        if not text:
            self.status_var.set("Prepare or enter a question first.")
            return
        question = dict(self.pending_question or {"id": f"manual_{now_iso()}", "phase": "manual"})
        question["text"] = text
        self.qa_send.config(state="disabled")
        self.status_var.set("Answering and adjudicating…")

        def worker():
            result = self.controller.answer_question(question)
            self.after(0, lambda: self.finish_question(result))
        threading.Thread(target=worker, daemon=True).start()

    def finish_question(self, result):
        self.qa_send.config(state="normal")
        self.qa_answer.delete("1.0", "end")
        self.qa_answer.insert("end", result.answer.get("answer_text", ""))
        self.status_var.set(f"{result.decision.title()}: {result.error or result.verdict.get('reasoning', '')}")
        self.refresh_concept()
        self.refresh_analyzer()
        self.refresh_library()

    def learn_current(self):
        if self.current_concept:
            self.start_learning([self.current_concept])

    def learn_selected(self):
        names = self.selected_concepts()

        if not names:
            self.status_var.set(
                "Select at least one concept."
            )
            self.learning_status_var.set(
                "No concepts selected."
            )
            self.learning_status.configure(
                foreground="#b45309"
            )
            return

        self.start_learning(names)

    def start_learning(self, concepts):
        if self.worker and self.worker.is_alive():
            self.status_var.set(
                "Learning is already running."
            )
            self.learning_status_var.set(
                "Learning already running…"
            )
            self.learning_status.configure(
                foreground="#2563eb"
            )
            return

        concepts = list(concepts)

        if not concepts:
            self.learning_status_var.set(
                "No concepts selected."
            )
            self.learning_status.configure(
                foreground="#b45309"
            )
            return

        self.stop_event.clear()

        self.progress.configure(
            maximum=len(concepts),
            value=0,
        )

        self.learning_status_var.set(
            f"Starting learning for {len(concepts)} "
            f"concept{'s' if len(concepts) != 1 else ''}…"
        )
        self.learning_status.configure(
            foreground="#2563eb"
        )

        self.status_var.set(
            "Learning started."
        )

        def worker():
            for index, concept in enumerate(
                    concepts,
                    start=1,
            ):
                if self.stop_event.is_set():
                    break

                self.after(
                    0,
                    lambda name=concept,
                           position=index,
                           total=len(concepts):
                    self.learning_status_var.set(
                        f"Learning {name} "
                        f"({position}/{total})…"
                    ),
                )

                controller = LearningController(
                    concept,
                    event_sink=self.engine_event,
                )

                self.running_controller = controller
                controller.learn_all()

                self.after(
                    0,
                    lambda value=index:
                    self.progress.configure(
                        value=value
                    ),
                )

            self.after(
                0,
                self.finish_learning,
            )

        self.worker = threading.Thread(
            target=worker,
            daemon=True,
        )
        self.worker.start()

    def finish_learning(self):
        self.running_controller = None

        if self.stop_event.is_set():
            message = "Learning stopped."
            colour = "#b45309"
        else:
            message = "Learning completed."
            colour = "#166534"

        self.status_var.set(message)
        self.learning_status_var.set(message)
        self.learning_status.configure(
            foreground=colour
        )

        self.refresh_library()
        self.refresh_concept()
        self.refresh_analyzer()

    def stop_learning(self):
        running = bool(
            self.worker
            and self.worker.is_alive()
        )

        self.stop_event.set()

        if self.running_controller:
            self.running_controller.stop()

        if running:
            self.learning_status_var.set(
                "Stopping learning…"
            )
            self.learning_status.configure(
                foreground="#b45309"
            )
            self.status_var.set(
                "Stopping learning…"
            )
        else:
            self.learning_status_var.set(
                "Learning idle."
            )
            self.learning_status.configure(
                foreground="#64748b"
            )

    # Analyzer ------------------------------------------------------------
    @staticmethod
    def answer_text(question):
        if question.get("answer"):
            return str(question["answer"])
        for value in reversed(question.get("answers") or []):
            text = value.get("text") if isinstance(value, dict) else value
            if text:
                return str(text)
        return ""

    def refresh_analyzer(self):
        self.analyzer_list.delete(0, "end")
        self.analyzer_questions = []
        record = find_concept(self.current_concept or "")
        if not record:
            return
        self.analyzer_questions = record.data.get("discovery", {}).get("questions", [])
        for index, question in enumerate(self.analyzer_questions, 1):
            saved = question.get("constituent_analysis") or {}
            marker = "✓" if saved.get("schema_version") == 2 else ("↻" if saved else "·")
            self.analyzer_list.insert("end", f"{index:02d} {marker} {question.get('text', '')}")

    @staticmethod
    def format_constituents(analysis):
        if not isinstance(analysis, dict):
            return ""
        labels = (
            ("Core meaning", "core_meaning"),
            ("Nouns", "nouns"),
            ("Verbs", "verbs"),
            ("Adjectives", "adjectives"),
            ("Adverbs", "adverbs"),
            ("Noun phrases", "noun_phrases"),
            ("Verb phrases", "verb_phrases"),
            ("Qualifiers", "qualifiers"),
            ("Relations", "relations"),
        )
        lines = []
        for label, key in labels:
            value = analysis.get(key)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value if item)
            if value:
                lines.append(f"{label}: {value}")
        return "\n\n".join(lines)

    def select_analysis_question(self, _event=None):
        selection = self.analyzer_list.curselection()
        if not selection:
            return
        question = self.analyzer_questions[selection[0]]
        saved = question.get("constituent_analysis") or {}
        analysis = saved.get("result", saved)
        values = (
            (self.analyzer_question, question.get("text", "")),
            (self.analyzer_answer, self.answer_text(question)),
            (self.analyzer_constituents, self.format_constituents(analysis)),
            (self.analyzer_json, json.dumps(saved, indent=2, ensure_ascii=False)),
        )
        for widget, value in values:
            widget.delete("1.0", "end")
            widget.insert("end", value)

    def analyze_selected(self):
        selection = self.analyzer_list.curselection()
        if not selection:
            self.analyzer_status.config(text="Select a question.", foreground="red")
            return
        question = self.analyzer_questions[selection[0]]
        answer = self.answer_text(question)
        if not answer:
            self.analyzer_status.config(text="The question has no accepted answer.", foreground="red")
            return
        record = find_concept(self.current_concept or "")
        if not record:
            self.analyzer_status.config(text="The concept file no longer exists.", foreground="red")
            return
        self.analyze_button.config(state="disabled")
        self.analyzer_status.config(text="Analyzing with API…", foreground="#333")
        threading.Thread(target=self.analysis_worker, args=(record.path, question, answer), daemon=True).start()

    def analysis_worker(self, record_path, question, answer):
        try:
            analysis, response = ConstituentAnalysisService().analyze(question.get("text", ""), answer)
            self.save_analysis(record_path, question, analysis, response)
            self.after(0, lambda: self.finish_analysis(analysis, None))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self.finish_analysis(None, e))

    def enqueue_analysis_spawns(self, analysis: dict) -> list[str]:
        """
        Extract lexical concept candidates from saved LLM constituents and add
        them to the existing spawn queue.

        Multiword constituents are split into individual lexical candidates,
        matching the original Analyzer spawn behaviour.
        """

        if not isinstance(analysis, dict):
            return []

        stopwords = {
            "a", "an", "the", "and", "or", "of", "in", "on", "at", "by",
            "for", "with", "from", "to", "as", "be", "is", "are", "was",
            "were", "been", "being", "it", "its", "this", "that", "these",
            "those", "their", "there", "here", "then", "when", "where",
            "how", "why", "which", "what", "who", "whom", "whose", "do",
            "does", "did", "doing", "have", "has", "had", "will", "would",
            "can", "could", "may", "might", "should", "shall", "must", "if",
            "because", "although", "though", "while", "so", "but", "than",
            "yet", "not", "no", "never", "also", "just", "even", "still",
            "already", "only", "all", "some", "any", "each", "every",
        }

        source_categories = (
            "nouns",
            "adjectives",
            "noun_phrases",
        )

        current_concept = normalize_concept_name(
            self.current_concept or ""
        )

        existing_concepts = {
            normalize_concept_name(record.name)
            for record in discover_concepts()
        }

        candidates = set()

        for category in source_categories:
            values = analysis.get(category, [])

            if not isinstance(values, list):
                continue

            for value in values:
                if not isinstance(value, str):
                    continue

                # LLM constituents are often phrases such as "citrus fruit".
                # Extract their individual concept-bearing words instead of
                # discarding the entire phrase.
                words = re.findall(
                    r"[A-Za-z][A-Za-z'-]*",
                    value.casefold(),
                )

                for word in words:
                    candidate = normalize_concept_name(word)

                    if (
                            not candidate
                            or len(candidate) <= 2
                            or candidate in stopwords
                            or candidate == current_concept
                            or candidate in existing_concepts
                            or not re.fullmatch(r"[a-z][a-z-]*", candidate)
                    ):
                        continue

                    candidates.add(candidate)

        if not candidates:
            return []

        existing_queue = load_spawn_queue()

        existing_keys = {
            value.casefold()
            for value in existing_queue
        }

        updated_queue = save_spawn_queue([
            *existing_queue,
            *sorted(candidates),
        ])

        return [
            value
            for value in updated_queue
            if value.casefold() not in existing_keys
        ]

    def save_analysis(
            self,
            record_path,
            question,
            analysis,
            response,
    ):
        payload = {
            "schema_version": 2,
            "source": "llm",
            "model": response.model,
            "created_at": now_iso(),
            "usage": response.usage,
            "result": analysis,
        }

        # Save constituent analysis into the same concept template.
        save_question_constituents(
            record_path,
            question.get("id"),
            payload,
        )

        # Restore the original Analyzer → spawn queue connection.
        return self.enqueue_analysis_spawns(analysis)

    def finish_analysis(self, analysis, error):
        self.analyze_button.config(state="normal")

        if error:
            self.analyzer_status.config(
                text=f"Analysis error: {error}",
                foreground="red",
            )
            return

        self.analyzer_constituents.delete("1.0", "end")
        self.analyzer_constituents.insert(
            "end",
            self.format_constituents(analysis),
        )

        self.analyzer_json.delete("1.0", "end")
        self.analyzer_json.insert(
            "end",
            json.dumps(
                analysis,
                indent=2,
                ensure_ascii=False,
            ),
        )

        self.analyzer_status.config(
            text="Analysis saved; spawn candidates updated.",
            foreground="#00695c",
        )

        self.refresh_analyzer()
        self.refresh_concept()
        self.refresh_spawns()

    def analyze_all(self):
        answered_questions = [
            (question, self.answer_text(question))
            for question in self.analyzer_questions
            if self.answer_text(question)
        ]

        if not answered_questions:
            self.analyzer_status.config(
                text="No accepted answers are available.",
                foreground="red",
            )
            return

        record = find_concept(self.current_concept or "")

        if not record:
            self.analyzer_status.config(
                text="The concept file no longer exists.",
                foreground="red",
            )
            return

        self.analyzer_status.config(
            text="Processing saved and unanswered analyses…",
            foreground="#333",
        )

        def worker():
            analyzed_count = 0
            reused_count = 0
            error = None
            service = None

            for question, answer in answered_questions:
                saved = question.get("constituent_analysis") or {}
                saved_result = saved.get("result")

                # Existing analyses must still repopulate spawn.json. Previously
                # they were skipped completely, which disconnected old results
                # from the Spawns tab.
                if (
                        saved.get("schema_version") == 2
                        and isinstance(saved_result, dict)
                ):
                    try:
                        self.enqueue_analysis_spawns(saved_result)
                        reused_count += 1
                    except Exception as exc:
                        error = str(exc)
                        break

                    continue

                try:
                    if service is None:
                        service = ConstituentAnalysisService()

                    analysis, response = service.analyze(
                        question.get("text", ""),
                        answer,
                    )

                    self.save_analysis(
                        record.path,
                        question,
                        analysis,
                        response,
                    )

                    analyzed_count += 1

                except Exception as exc:
                    error = str(exc)
                    break

            self.after(
                0,
                lambda analyzed=analyzed_count,
                       reused=reused_count,
                       failure=error: self.finish_analyze_all(
                    analyzed,
                    reused,
                    failure,
                ),
            )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def finish_analyze_all(
            self,
            analyzed_count,
            reused_count,
            error,
    ):
        if error:
            message = (
                f"Analyzed {analyzed_count}; "
                f"reused {reused_count} saved analyses. "
                f"Error: {error}"
            )
        else:
            message = (
                f"Analyzed {analyzed_count}; "
                f"reused {reused_count} saved analyses. "
                "Spawn candidates updated."
            )

        self.analyzer_status.config(
            text=message,
            foreground="red" if error else "#00695c",
        )

        self.refresh_analyzer()
        self.refresh_concept()
        self.refresh_spawns()

    # Spawns --------------------------------------------------------------
    def refresh_spawns(self):
        self.spawn_list.delete(0, "end")
        values = self.spawn_service.candidates()
        for value in values:
            self.spawn_list.insert("end", value)
        self.spawn_status.config(text=f"{len(values)} candidate(s).")

    def edit_selected_spawn(self, _event=None):
        """
        Edit exactly one spawn candidate without applying automatic
        normalization to the user's correction.
        """

        selection = self.spawn_list.curselection()

        if len(selection) != 1:
            self.spawn_status.config(
                text="Select exactly one candidate to edit.",
                foreground="#b45309",
            )
            return

        index = selection[0]
        original = self.spawn_list.get(index)

        corrected = simpledialog.askstring(
            "Edit spawn candidate",
            "Correct the concept name:",
            initialvalue=original,
            parent=self,
        )

        if corrected is None:
            return

        corrected = " ".join(
            corrected.strip().split()
        )

        if not corrected:
            self.spawn_status.config(
                text="The candidate name cannot be empty.",
                foreground="red",
            )
            return

        queue = load_spawn_queue()
        updated = []
        replaced = False

        for candidate in queue:
            if (
                    not replaced
                    and candidate.casefold() == original.casefold()
            ):
                updated.append(corrected)
                replaced = True
            else:
                updated.append(candidate)

        if not replaced:
            self.spawn_status.config(
                text="The selected candidate is no longer in the queue.",
                foreground="red",
            )
            self.refresh_spawns()
            return

        save_spawn_queue(updated)
        self.refresh_spawns()

        # Restore selection to the corrected item.
        for new_index in range(self.spawn_list.size()):
            if (
                    self.spawn_list.get(new_index).casefold()
                    == corrected.casefold()
            ):
                self.spawn_list.selection_clear(
                    0,
                    "end",
                )
                self.spawn_list.selection_set(
                    new_index,
                )
                self.spawn_list.see(
                    new_index,
                )
                break

        self.spawn_status.config(
            text=f"Changed '{original}' to '{corrected}'.",
            foreground="#166534",
        )

    def remove_spawns(self):
        values = [self.spawn_list.get(index) for index in self.spawn_list.curselection()]
        self.spawn_service.remove(values)
        self.refresh_spawns()

    def auto_spawn(self):
        created = self.spawn_service.spawn_all()
        self.spawn_status.config(text=f"Spawned {len(created)} concept(s).")
        self.refresh_spawns()
        self.refresh_library(created)

    # Stage 2 -------------------------------------------------------------
    def open_model_stage(self):
        if self.model_stage_window and self.model_stage_window.winfo_exists():
            self.model_stage_window.refresh_knowledge()
            self.model_stage_window.deiconify()
            self.model_stage_window.lift()
            self.model_stage_window.focus_force()
            return
        self.model_stage_window = ModelStageWindow(self, on_closed=self.model_stage_closed)

    def model_stage_closed(self):
        self.model_stage_window = None

    def on_close(self):
        self.stop_learning()
        if self.model_stage_window and self.model_stage_window.winfo_exists():
            self.model_stage_window.destroy()
        self.destroy()
