"""The desktop interface. Every widget and background operation is Python.

Tk is touched only on the main thread. Workers return values through a Queue;
the event loop handles rendering, errors and progress without freezing the UI.
"""
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from .knowledge import read_document
from .ollama_client import OllamaClient
from .settings import load_settings, save_settings

BG = "#F3F5F8"
WHITE = "#FFFFFF"
INK = "#152D3C"
MUTED = "#627785"
TEAL = "#007D76"
PALE = "#DDF2EE"
BORDER = "#DAE3E9"
NAV = "#152D3C"
COLORS = {"CS": "#137D9D", "Math": "#845EC2", "Food": "#CF8227", "Sports": "#25966D"}


def fill_text(widget, text):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", str(text))
    widget.configure(state="disabled")


class Application:
    def __init__(self, root, store, lab, data_dir):
        self.root, self.store, self.lab = root, store, lab
        self.data_dir = Path(data_dir)
        self.settings = load_settings(self.data_dir)
        self.work_queue = queue.Queue()
        self.busy = False
        self.task_buttons = []
        self.sources = []
        self.last_answer = ""
        self.highlight_ids = set()
        root.title("Rax Ai | Python Knowledge Studio")
        root.geometry("1220x820")
        root.minsize(1020, 740)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._shell()
        self._home()
        self._documents()
        self._ask()
        self._vectors()
        self._settings()
        if not self.settings["samples_initialized"]:
            self.store.load_samples()
            self.settings["samples_initialized"] = True
            save_settings(self.data_dir, self.settings)
        self.refresh_documents()
        self.show_page("Home")
        root.after(80, self._poll)

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.option_add("*Font", ("Segoe UI", 11))
        style.configure(".", font=("Segoe UI", 11), background=BG, foreground=INK)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=WHITE)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Card.TLabel", background=WHITE)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("CardMuted.TLabel", foreground=MUTED, background=WHITE)
        style.configure("Heading.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("CardTitle.TLabel", font=("Segoe UI", 14, "bold"), background=WHITE)
        style.configure("TButton", padding=(13, 9), background=WHITE, bordercolor=BORDER,
                        lightcolor=WHITE, darkcolor=WHITE, focuscolor=BORDER)
        style.map("TButton", background=[("active", PALE), ("disabled", "#E7EDF0")])
        style.configure("Primary.TButton", background=TEAL, foreground=WHITE, bordercolor=TEAL)
        style.map("Primary.TButton", background=[("active", "#006960"), ("disabled", "#87B5B1")],
                  foreground=[("disabled", WHITE)])
        style.configure("TEntry", fieldbackground=WHITE, padding=7)
        style.configure("TCombobox", fieldbackground=WHITE, padding=7)
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, rowheight=32, borderwidth=0)
        style.configure("Treeview.Heading", background="#EAF0F4", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Treeview", background=[("selected", TEAL)], foreground=[("selected", WHITE)])
        style.configure("Horizontal.TProgressbar", background=TEAL, troughcolor=BG, borderwidth=0)

    def _shell(self):
        self.sidebar = tk.Frame(self.root, bg=NAV, width=200)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="YOUR OWN\nAI", bg=NAV, fg=WHITE, justify="left",
                 font=("Segoe UI", 23, "bold")).pack(anchor="w", padx=24, pady=(32, 8))
        tk.Label(self.sidebar, text="PYTHON KNOWLEDGE STUDIO", bg=NAV, fg="#86C7BE",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=24, pady=(0, 32))
        self.nav_buttons = {}
        self.pages = {}
        names = ["Home", "Documents", "Ask documents", "Vector lab", "Settings"]
        for i, name in enumerate(names):
            button = tk.Button(self.sidebar, text=f"{i+1:02d}   {name}", anchor="w", padx=20,
                               pady=13, relief="flat", bd=0, bg=NAV, fg="#CAD9E2",
                               activebackground="#244B5B", activeforeground=WHITE,
                               cursor="hand2", command=lambda n=name: self.show_page(n))
            button.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[name] = button
        tk.Label(self.sidebar, text="LOCAL FIRST\nYour files. Your computer.", justify="left", bg=NAV,
                 fg="#9BB3BF", font=("Segoe UI", 10)).pack(side="bottom", anchor="w", padx=24, pady=28)
        shell = ttk.Frame(self.root)
        shell.pack(side="left", fill="both", expand=True)
        footer = ttk.Frame(shell)
        footer.pack(side="bottom", fill="x", padx=26, pady=(4, 14))
        self.status = tk.StringVar(value="Ready. Offline mode needs no models or API key.")
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=115)
        self.progress.pack(side="right")
        self.content = ttk.Frame(shell, padding=(26, 23, 26, 8))
        self.content.pack(fill="both", expand=True)
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)
        for name in names:
            frame = ttk.Frame(self.content)
            frame.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = frame

    def show_page(self, name):
        self.pages[name].tkraise()
        self.current_page = name
        for key, button in self.nav_buttons.items():
            button.configure(bg=TEAL if key == name else NAV, fg=WHITE if key == name else "#CAD9E2")
        if name == "Vector lab":
            self.root.after_idle(self.draw_plot)

    def _heading(self, parent, title, subtitle):
        ttk.Label(parent, text=title, style="Heading.TLabel").pack(anchor="w")
        ttk.Label(parent, text=subtitle, style="Muted.TLabel", wraplength=700).pack(anchor="w", pady=(7, 20))

    def _button(self, parent, text, command, primary=False, **pack):
        button = ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton")
        button.pack(**pack)
        self.task_buttons.append(button)
        return button

    def _text(self, parent, height=10, editable=False):
        widget = ScrolledText(parent, height=height, wrap="word", bg=WHITE, fg=INK,
                              relief="flat", bd=0, padx=15, pady=12, insertbackground=TEAL,
                              font=("Segoe UI", 11), spacing1=3, spacing3=5)
        if not editable:
            widget.configure(state="disabled")
        return widget

    def _tree(self, parent, columns, heights=8):
        box = ttk.Frame(parent)
        box.pack(fill="both", expand=True)
        tree = ttk.Treeview(box, columns=[x[0] for x in columns], show="headings", height=heights, selectmode="browse")
        for name, title, width in columns:
            tree.heading(name, text=title)
            tree.column(name, width=width, minwidth=45, stretch=True)
        scroll = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        return tree

    def _home(self):
        frame = self.pages["Home"]
        self._heading(frame, "Your knowledge, connected.", "A simple workspace to explore vector search and ask questions about your own documents.")
        hero = tk.Frame(frame, bg=PALE, padx=22, pady=18)
        hero.pack(fill="x", pady=(0, 20))
        tk.Label(hero, text="Start with an answer.\nThen explore how it works.", bg=PALE, fg=INK,
                 font=("Segoe UI", 22, "bold"), justify="left").pack(anchor="w")
        tk.Label(hero, text="Sample documents are ready. No account, API key or model download is needed for offline search.",
                 bg=PALE, fg=MUTED, wraplength=680, justify="left").pack(anchor="w", pady=(10, 12))
        actions = tk.Frame(hero, bg=PALE)
        actions.pack(anchor="w")
        self._button(actions, "Ask a sample question", self.sample_question, True, side="left", padx=(0, 10))
        self._button(actions, "Explore vector lab", lambda: self.show_page("Vector lab"), side="left")
        stats = ttk.Frame(frame)
        stats.pack(fill="x", pady=(0, 20))
        self.stat_vars = {}
        for i, (key, label) in enumerate([("documents", "SAVED DOCUMENTS"), ("chunks", "SEARCHABLE PASSAGES"), ("vectors", "DEMO VECTORS")]):
            card = ttk.Frame(stats, style="Card.TFrame", padding=16)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            stats.columnconfigure(i, weight=1)
            self.stat_vars[key] = tk.StringVar(value="0")
            ttk.Label(card, textvariable=self.stat_vars[key], style="Card.TLabel", font=("Segoe UI", 25, "bold")).pack(anchor="w")
            ttk.Label(card, text=label, style="CardMuted.TLabel", font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))
        steps = ttk.Frame(frame, style="Card.TFrame", padding=16)
        steps.pack(fill="x")
        ttk.Label(steps, text="Two ways to learn", style="CardTitle.TLabel").pack(anchor="w")
        for title, detail in [
            ("01   Your documents", "Paste notes or import a file. Ask a question and inspect its numbered sources."),
            ("02   The search engine", "Compare three algorithms, view 16D demo vectors in 2D, and measure recall."),
        ]:
            ttk.Label(steps, text=title, style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 3))
            ttk.Label(steps, text=detail, style="CardMuted.TLabel", wraplength=700).pack(anchor="w")
        self._button(frame, "Load sample documents", self.load_samples, side="left", pady=12)
        ttk.Label(frame, text="Generated answers are optional: connect Ollama in Settings.", style="Muted.TLabel", wraplength=470).pack(side="left", padx=15, pady=12)

    def _documents(self):
        frame = self.pages["Documents"]
        self._heading(frame, "Documents", "Add your notes once. Documents are saved on this computer and split into searchable passages.")
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(0, 15))
        for title, command, primary in [("Add document", self.add_document_dialog, True), ("Import file", self.import_file, False),
                                          ("Export backup", self.export_backup, False), ("Import backup", self.import_backup, False)]:
            self._button(toolbar, title, command, primary, side="left", padx=(0, 8))
        split = ttk.Panedwindow(frame, orient="horizontal")
        split.pack(fill="both", expand=True)
        left = ttk.Frame(split)
        right = ttk.Frame(split, style="Card.TFrame", padding=12)
        split.add(left, weight=2)
        split.add(right, weight=3)
        self.doc_tree = self._tree(left, [("title", "Document", 230), ("chunks", "Passages", 75)], 13)
        self.doc_tree.bind("<<TreeviewSelect>>", self.select_document)
        self.doc_title = tk.StringVar(value="Select a document to read")
        ttk.Label(right, textvariable=self.doc_title, style="CardTitle.TLabel", wraplength=430).pack(anchor="w", padx=8, pady=6)
        self.doc_text = self._text(right)
        self.doc_text.pack(fill="both", expand=True)
        fill_text(self.doc_text, "Choose a document on the left, or add your own notes.\n\nSupported: UTF-8 TXT and Markdown. Text-based PDF import is available with the optional pypdf package.")
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(14, 0))
        self._button(bar, "Delete selected", self.delete_document, side="left")
        self.doc_count = tk.StringVar()
        ttk.Label(bar, textvariable=self.doc_count, style="Muted.TLabel").pack(side="left", padx=16)

    def refresh_documents(self):
        selected = self.doc_tree.selection()
        self.doc_tree.delete(*self.doc_tree.get_children())
        for doc in self.store.list_documents():
            self.doc_tree.insert("", "end", iid=str(doc["id"]), values=(doc["title"], doc["chunks"]))
        if selected and self.doc_tree.exists(selected[0]):
            self.doc_tree.selection_set(selected[0])
        stats = self.store.stats()
        for key in ("documents", "chunks"):
            self.stat_vars[key].set(str(stats[key]))
        self.stat_vars["vectors"].set(str(len(self.lab.items)))
        self.doc_count.set(f'{stats["documents"]} documents  |  {stats["chunks"]} passages  |  Saved automatically')

    def select_document(self, event=None):
        selected = self.doc_tree.selection()
        if selected:
            document = self.store.get_document(selected[0])
            if document is None:
                self.refresh_documents()
                self.doc_title.set("Select a document to read")
                fill_text(self.doc_text, "This document is no longer available. Choose another document.")
                return
            self.doc_title.set(document["title"])
            fill_text(self.doc_text, document["text"])

    def add_document_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Add a document")
        dialog.geometry("720x570")
        dialog.minsize(540, 420)
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        box = ttk.Frame(dialog, padding=22)
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="Document title").pack(anchor="w")
        title = ttk.Entry(box)
        title.pack(fill="x", pady=(6, 15))
        ttk.Label(box, text="Paste your notes or document text").pack(anchor="w")
        content = self._text(box, editable=True)
        content.pack(fill="both", expand=True, pady=(6, 15))
        def save():
            name, text = title.get(), content.get("1.0", "end-1c")
            def done(result):
                if dialog.winfo_exists():
                    dialog.destroy()
                self.refresh_documents()
                self.show_page("Documents")
                self.doc_tree.selection_set(str(result))
                self.select_document()
            self.run_task("Saving document", lambda: self.store.add_document(name, text), done)
        self._button(box, "Save document", save, True, side="right")
        ttk.Label(box, text="Up to 200,000 characters per document.", style="Muted.TLabel").pack(side="left")
        title.focus_set()

    def import_file(self):
        path = filedialog.askopenfilename(title="Import a document", filetypes=[("Documents", "*.txt *.md *.pdf"), ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            self.run_task("Reading and saving document", lambda: self.store.add_document(Path(path).stem, read_document(path)),
                          lambda result: self.refresh_documents())

    def delete_document(self):
        selected = self.doc_tree.selection()
        if not selected:
            messagebox.showinfo("Select a document", "Choose the document you want to delete.")
            return
        doc_id = selected[0]
        title = self.doc_tree.item(selected[0], "values")[0]
        if messagebox.askyesno("Delete document?", f'Delete "{title}" and its searchable passages?\n\nExport a backup first if you want to restore it later.'):
            def done(result):
                self.refresh_documents()
                self.doc_title.set("Select a document to read")
                fill_text(self.doc_text, "The document was deleted.")
            self.run_task("Deleting document", lambda: self.store.delete_document(doc_id), done)

    def load_samples(self):
        self.run_task("Loading sample documents", self.store.load_samples, lambda result: self.refresh_documents())

    def export_backup(self):
        path = filedialog.asksaveasfilename(title="Export document backup", defaultextension=".json", initialfile="your_ai_backup.json", filetypes=[("JSON backup", "*.json")])
        if path:
            self.run_task("Exporting backup", lambda: self.store.export_backup(path), lambda result: messagebox.showinfo("Backup saved", f"Your documents were saved to:\n{path}"))

    def import_backup(self):
        path = filedialog.askopenfilename(title="Import document backup", filetypes=[("JSON backup", "*.json")])
        if path and messagebox.askyesno("Import backup?", "Merge documents from this backup into your collection?\n\nYour existing documents are preserved."):
            self.run_task("Importing backup", lambda: self.store.import_backup(path), lambda result: self.refresh_documents())

    def _ask(self):
        frame = self.pages["Ask documents"]
        self._heading(frame, "Ask your documents", "Find useful passages, read an answer, and check the source. Every question searches your saved collection.")
        card = ttk.Frame(frame, style="Card.TFrame", padding=18)
        card.pack(fill="x", pady=(0, 16))
        ttk.Label(card, text="What would you like to know?", style="CardTitle.TLabel").pack(anchor="w")
        self.question = tk.StringVar()
        entry = ttk.Entry(card, textvariable=self.question)
        entry.pack(fill="x", pady=12)
        entry.bind("<Return>", lambda event: self.ask_question())
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x")
        self.ask_mode = tk.StringVar(value="Offline (exact extracts)")
        ttk.Combobox(row, textvariable=self.ask_mode, state="readonly", width=25,
                     values=["Offline (exact extracts)", "Ollama (local AI)"]).pack(side="left", padx=(0, 10))
        self._button(row, "Ask question", self.ask_question, True, side="left", padx=(0, 8))
        self._button(row, "Find passages", lambda: self.ask_question(search_only=True), side="left")
        split = ttk.Panedwindow(frame, orient="horizontal")
        split.pack(fill="both", expand=True)
        answer_card = ttk.Frame(split, style="Card.TFrame", padding=12)
        sources_card = ttk.Frame(split, style="Card.TFrame", padding=12)
        split.add(answer_card, weight=3)
        split.add(sources_card, weight=2)
        self.answer_label = tk.StringVar(value="Your answer")
        ttk.Label(answer_card, textvariable=self.answer_label, style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=6)
        self.answer_text = self._text(answer_card)
        self.answer_text.pack(fill="both", expand=True)
        fill_text(self.answer_text, "Start with a question about your notes.\n\nOffline mode returns exact excerpts from the most relevant passages. It does not generate new text.\n\nChoose Ollama for a generated answer after installing and connecting the local models in Settings.")
        ttk.Label(sources_card, text="Sources", style="CardTitle.TLabel").pack(anchor="w", padx=8, pady=6)
        self.source_tree = self._tree(sources_card, [("ref", "Ref", 45), ("title", "Document", 180), ("score", "Score", 65)], 4)
        self.source_tree.bind("<<TreeviewSelect>>", self.select_source)
        self.source_text = self._text(sources_card, height=9)
        self.source_text.pack(fill="both", expand=True, pady=(10, 0))
        fill_text(self.source_text, "Select a source to inspect the exact retrieved passage. A relevance score is not a confidence percentage.")
        bottom = ttk.Frame(frame)
        bottom.pack(fill="x", pady=(12, 0))
        self._button(bottom, "Export answer", self.export_answer, side="left")
        ttk.Label(bottom, text="Review source passages before relying on generated answers.", style="Muted.TLabel").pack(side="left", padx=15)

    def client(self):
        return OllamaClient(base_url=self.settings["ollama_url"], embedding_model=self.settings["embedding_model"],
                            generation_model=self.settings["generation_model"])

    def ask_question(self, search_only=False):
        question = self.question.get().strip()
        mode = "ollama" if self.ask_mode.get().startswith("Ollama") else "offline"
        if not question:
            messagebox.showinfo("Add a question", "Type a question about a saved document first.")
            return
        try:
            client = self.client() if mode == "ollama" else None
        except ValueError as exc:
            messagebox.showerror("Invalid local AI settings", f"{exc}\n\nOpen Settings to correct and save the local AI configuration.")
            return
        def work():
            if search_only:
                sources = self.store.search(question, k=4, mode=mode, client=client)
                answer = "Retrieved passages:\n\n" + "\n\n".join(f'[{i}] {s["text"]}' for i, s in enumerate(sources, 1)) if sources else "No matching passages found. Try words that appear in your documents."
                return {"sources": sources, "answer": answer, "mode": mode}
            return self.store.ask(question, k=4, mode=mode, client=client)
        def done(result):
            self.sources = result["sources"]
            self.last_answer = f"Question: {question}\nMode: {mode}\n\n{result['answer']}\n\nSources:\n" + "\n\n".join(f'[{i}] {s["title"]}\n{s["text"]}' for i, s in enumerate(self.sources, 1))
            fill_text(self.answer_text, result["answer"])
            self.answer_label.set("Matching passages" if search_only else ("Local AI answer" if mode == "ollama" else "Offline answer - exact extracts"))
            self.source_tree.delete(*self.source_tree.get_children())
            for i, source in enumerate(self.sources):
                self.source_tree.insert("", "end", iid=str(i), values=(f"[{i+1}]", source["title"], f'{source["score"]:.3f}'))
            if self.sources:
                self.source_tree.selection_set("0")
                self.select_source()
            else:
                fill_text(self.source_text, "No source matched this question.")
        self.run_task("Searching documents" if mode == "offline" else "Running local AI; first use may take a few minutes", work, done)

    def sample_question(self):
        self.question.set("What is RAG?")
        self.ask_mode.set("Offline (exact extracts)")
        self.show_page("Ask documents")
        self.ask_question()

    def select_source(self, event=None):
        selected = self.source_tree.selection()
        if selected:
            source = self.sources[int(selected[0])]
            fill_text(self.source_text, f'[{int(selected[0]) + 1}] {source["title"]}\n\n{source["text"]}')

    def export_answer(self):
        if not self.last_answer:
            messagebox.showinfo("No answer yet", "Ask a question first, then export the answer and its sources.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="your_ai_answer.txt", filetypes=[("Text file", "*.txt")])
        if path:
            try:
                Path(path).write_text(self.last_answer, encoding="utf-8")
                self.status.set("Answer and sources exported.")
            except OSError as exc:
                messagebox.showerror("Cannot save answer", str(exc))

    def _vectors(self):
        frame = self.pages["Vector lab"]
        self._heading(frame, "Vector search lab", "Compare three real Python search algorithms on synthetic demo vectors. Lower distance means closer.")
        controls = ttk.Frame(frame, style="Card.TFrame", padding=16)
        controls.pack(fill="x", pady=(0, 14))
        row = ttk.Frame(controls, style="Card.TFrame")
        row.pack(fill="x")
        self.vector_query = tk.StringVar(value=self.lab.items[0]["label"])
        self.algorithm = tk.StringVar(value="hnsw")
        self.metric = tk.StringVar(value="cosine")
        self.top_k = tk.StringVar(value="5")
        for name, var, values, width in [
            ("Demo query", self.vector_query, [x["label"] for x in self.lab.items], 27),
            ("Algorithm", self.algorithm, ["hnsw", "kd_tree", "brute_force"], 13),
            ("Distance", self.metric, ["cosine", "euclidean", "manhattan"], 12),
            ("Top K", self.top_k, ["3", "5", "10", "20"], 6),
        ]:
            group = ttk.Frame(row, style="Card.TFrame")
            group.pack(side="left", fill="x", expand=name == "Demo query", padx=(0, 12))
            ttk.Label(group, text=name, style="CardMuted.TLabel", font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 5))
            combo = ttk.Combobox(group, textvariable=var, values=values, state="normal" if name == "Demo query" else "readonly", width=width)
            combo.pack(fill="x")
            if name == "Demo query":
                self.query_combo = combo
                combo.bind("<Return>", lambda event: self.search_vectors())
        buttons = ttk.Frame(controls, style="Card.TFrame")
        buttons.pack(fill="x", pady=(12, 0))
        self._button(buttons, "Search vectors", self.search_vectors, True, side="left", padx=(0, 8))
        self._button(buttons, "Compare algorithms", self.compare_vectors, side="left", padx=(0, 8))
        self._button(buttons, "Manage vectors", self.manage_vectors, side="left")
        split = ttk.Panedwindow(frame, orient="horizontal")
        split.pack(fill="both", expand=True)
        left = ttk.Frame(split, style="Card.TFrame", padding=12)
        right = ttk.Frame(split, style="Card.TFrame", padding=12)
        split.add(left, weight=1)
        split.add(right, weight=1)
        self.result_label = tk.StringVar(value="Nearest matches")
        ttk.Label(left, textvariable=self.result_label, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))
        self.result_tree = self._tree(left, [("label", "Match", 170), ("category", "Category", 80), ("distance", "Distance", 80)], 6)
        ttk.Label(right, text="Vector map - 2D PCA", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 5))
        self.plot = tk.Canvas(right, bg=WHITE, highlightthickness=0, width=350, height=230)
        self.plot.pack(fill="both", expand=True)
        self.plot.bind("<Configure>", lambda event: self.draw_plot())
        self.plot_caption = tk.StringVar(value="Click a point to use it as the query. Ringed points are search results.")
        ttk.Label(right, textvariable=self.plot_caption, style="CardMuted.TLabel", wraplength=380, font=("Segoe UI", 9)).pack(anchor="w", pady=4)
        bench = ttk.Frame(frame, style="Card.TFrame", padding=(12, 10))
        bench.pack(fill="x", pady=(12, 0))
        ttk.Label(bench, text="Benchmark  /  measured search time and recall", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        self.benchmark_tree = self._tree(bench, [("algo", "Algorithm", 170), ("time", "Mean time (ms)", 140), ("recall", "Recall@K", 100)], 3)
        self.graph_note = tk.StringVar(value="Choose Compare algorithms to measure this query. Index build time is excluded.")
        ttk.Label(frame, textvariable=self.graph_note, style="Muted.TLabel", font=("Segoe UI", 9), wraplength=840).pack(anchor="w", pady=(9, 0))

    def _vector_inputs(self):
        query = self.vector_query.get().strip()
        if query.startswith("["):
            query = json.loads(query)
        return query, int(self.top_k.get()), self.metric.get()

    def search_vectors(self):
        try:
            query, k, metric = self._vector_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid vector query", str(exc))
            return
        algorithm = self.algorithm.get()
        self.run_task("Searching vectors", lambda: self.lab.search(query, k=k, metric=metric, algorithm=algorithm), self.show_vector_results)

    def show_vector_results(self, result):
        self.result_tree.delete(*self.result_tree.get_children())
        self.highlight_ids = {item["id"] for item in result["results"]}
        for item in result["results"]:
            self.result_tree.insert("", "end", values=(item["label"], item["category"], f'{item["distance"]:.5f}'))
        self.result_label.set(f'Nearest matches  /  {result["elapsed_ms"]:.3f} ms')
        self.draw_plot()

    def compare_vectors(self):
        try:
            query, k, metric = self._vector_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid vector query", str(exc))
            return
        def done(rows):
            self.benchmark_tree.delete(*self.benchmark_tree.get_children())
            for row in rows:
                self.benchmark_tree.insert("", "end", values=(row["algorithm"], f'{row["mean_ms"]:.4f}', f'{row["recall_at_k"]:.0%}'))
            self.graph_note.set("5 warmed searches per algorithm. Recall compares IDs with exact brute force. A small demo is not a scalability test.")
        self.run_task("Comparing search algorithms", lambda: self.lab.benchmark(query, k=k, metric=metric, repeats=5), done)

    def draw_plot(self):
        if not hasattr(self, "plot") or not self.plot.winfo_exists():
            return
        canvas = self.plot
        canvas.delete("all")
        points = self.lab.project()
        if not points:
            canvas.create_text(180, 90, text="No vectors. Add a vector or restart to reload demos.", fill=MUTED, width=280)
            return
        width, height = max(canvas.winfo_width(), 250), max(canvas.winfo_height(), 180)
        min_x, max_x = min(p["x"] for p in points), max(p["x"] for p in points)
        min_y, max_y = min(p["y"] for p in points), max(p["y"] for p in points)
        for fraction in (0.25, 0.5, 0.75):
            canvas.create_line(20, 25 + fraction * (height - 80), width - 20, 25 + fraction * (height - 80), fill="#EDF1F4")
        categories = sorted({p["category"] for p in points})
        for i, category in enumerate(categories[:6]):
            x = 16 + i % 3 * ((width - 30) / 3)
            y = height - 28 + (i // 3) * 15
            color = COLORS.get(category, "#697A85")
            canvas.create_oval(x, y, x + 7, y + 7, fill=color, outline="")
            canvas.create_text(x + 13, y + 3, text=category, anchor="w", fill=MUTED, font=("Segoe UI", 8))
        for point in points:
            x = 32 + (point["x"] - min_x) / (max_x - min_x or 1) * (width - 64)
            y = 25 + (max_y - point["y"]) / (max_y - min_y or 1) * (height - 85)
            color = COLORS.get(point["category"], "#697A85")
            if point["id"] in self.highlight_ids:
                canvas.create_oval(x - 9, y - 9, x + 9, y + 9, outline=TEAL, width=2)
            dot = canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline=WHITE, width=1)
            canvas.tag_bind(dot, "<Enter>", lambda event, p=point: self.plot_caption.set(f'{p["label"]}  |  {p["category"]}'))
            canvas.tag_bind(dot, "<Leave>", lambda event: self.plot_caption.set("Click a point to use it as the query. Rings mark search results."))
            canvas.tag_bind(dot, "<Button-1>", lambda event, p=point: self.vector_query.set(p["label"]))

    def manage_vectors(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage demo vectors")
        dialog.geometry("820x570")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        box = ttk.Frame(dialog, padding=20)
        box.pack(fill="both", expand=True)
        ttk.Label(box, text="Demo vectors are temporary. Restarting restores the original demo set.", style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
        tree = self._tree(box, [("id", "ID", 50), ("label", "Label", 240), ("category", "Category", 110)], 7)
        form = ttk.Frame(box)
        form.pack(fill="x", pady=12)
        label, category, vector = tk.StringVar(), tk.StringVar(value="Custom"), tk.StringVar()
        for title, var in [("Label", label), ("Category", category), ("16 numbers, separated by commas", vector)]:
            ttk.Label(form, text=title).pack(anchor="w", pady=(5, 2))
            ttk.Entry(form, textvariable=var).pack(fill="x")
        def refresh():
            if dialog.winfo_exists():
                tree.delete(*tree.get_children())
                for item in self.lab.items:
                    tree.insert("", "end", iid=str(item["id"]), values=(item["id"], item["label"], item["category"]))
            self.query_combo.configure(values=[x["label"] for x in self.lab.items])
            self.stat_vars["vectors"].set(str(len(self.lab.items)))
            self.highlight_ids.clear()
            self.result_tree.delete(*self.result_tree.get_children())
            self.benchmark_tree.delete(*self.benchmark_tree.get_children())
            self.draw_plot()
        def add():
            try:
                values = [float(x.strip()) for x in vector.get().split(",")]
            except ValueError:
                messagebox.showerror("Invalid vector", "Enter 16 numbers separated by commas.", parent=dialog)
                return
            name, group = label.get(), category.get()
            self.run_task("Adding vector and rebuilding indexes", lambda: self.lab.add(name, group, values), lambda result: refresh())
        def delete():
            selected = tree.selection()
            if selected:
                item_id = int(selected[0])
                self.run_task("Deleting vector and rebuilding indexes", lambda: self.lab.delete(item_id), lambda result: refresh())
        bar = ttk.Frame(box)
        bar.pack(fill="x")
        self._button(bar, "Add vector", add, True, side="right")
        self._button(bar, "Delete selected", delete, side="left")
        refresh()

    def _settings(self):
        frame = self.pages["Settings"]
        self._heading(frame, "Settings & quick help", "Offline mode is ready to use. Set up local AI only when you want semantic search and generated answers.")
        card = ttk.Frame(frame, style="Card.TFrame", padding=20)
        card.pack(fill="x")
        ttk.Label(card, text="Optional local AI / Ollama", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self.setting_vars = {}
        for i, (key, label) in enumerate([("ollama_url", "Local server URL"), ("embedding_model", "Embedding model"), ("generation_model", "Answer model")], 1):
            ttk.Label(card, text=label, style="Card.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 18), pady=7)
            var = tk.StringVar(value=self.settings[key])
            self.setting_vars[key] = var
            ttk.Entry(card, textvariable=var).grid(row=i, column=1, sticky="ew", pady=7)
        card.columnconfigure(1, weight=1)
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self._button(actions, "Save settings", self.save_preferences, True, side="left", padx=(0, 10))
        self._button(actions, "Check connection", self.check_ollama, side="left")
        self.connection_message = tk.StringVar(value="Not checked. Ollama is not needed for offline mode.")
        ttk.Label(card, textvariable=self.connection_message, style="CardMuted.TLabel", wraplength=790).grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))
        help_text = self._text(frame, height=13)
        help_text.pack(fill="both", expand=True, pady=(18, 0))
        fill_text(help_text,
            "QUICK START\n"
            "1. Documents: import TXT/MD or paste notes. Text PDF import needs optional pypdf.\n"
            "2. Ask documents: enter a question and inspect the numbered source passages.\n"
            "3. Vector lab: select a demo concept, search, then compare algorithms.\n\n"
            "ENABLE LOCAL AI (OPTIONAL)\n"
            "Install Ollama from https://ollama.com/download, then run in a terminal:\n"
            "    ollama pull nomic-embed-text\n"
            "    ollama pull llama3.2:1b\n"
            "Keep Ollama running, click Check connection, then select Ollama in Ask documents.\n"
            "Model downloads need internet and disk space. CPU generation can be slow.\n\n"
            f"YOUR SAVED DATA\n{self.data_dir}\n"
            "Document backups contain plain text. Store them somewhere you trust. Offline answers, "
            "vector lab changes and UI selections are not saved in backups; documents are.\n\n"
            "ABOUT THIS BUILD\n"
            "Python + Tkinter + SQLite. No browser, C++ compiler or mandatory pip package. "
            "The small vector lab is for learning, not production-scale performance. "
            "Offline mode matches words; it cannot understand every paraphrase. "
            "Local AI can make mistakes even when sources are supplied. "
            "Model weights are installed separately; use local models to keep processing on your machine.\n\n"
            "For setup, troubleshooting and the project presentation, open README.md and the docs folder.")

    def _draft_settings(self):
        candidate = self.settings.copy()
        for key, var in self.setting_vars.items():
            candidate[key] = var.get().strip()
        # Client construction validates local URL and model names without a request.
        OllamaClient(base_url=candidate["ollama_url"], embedding_model=candidate["embedding_model"], generation_model=candidate["generation_model"])
        return candidate

    def save_preferences(self):
        try:
            candidate = self._draft_settings()
            save_settings(self.data_dir, candidate)
            self.settings = candidate
            self.status.set("Settings saved. Select Ollama in Ask documents when the models are ready.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Cannot save settings", str(exc))

    def check_ollama(self):
        try:
            candidate = self._draft_settings()
            client = OllamaClient(base_url=candidate["ollama_url"], embedding_model=candidate["embedding_model"], generation_model=candidate["generation_model"], timeout=8)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        def done(result):
            names = result.get("models", [])
            if result.get("online"):
                text = "Connected. " + str(result.get("message", "")) + " Models: " + ", ".join(str(x) for x in names)
                self.connection_message.set(text[:380] + " | Save settings to apply changes.")
            else:
                self.connection_message.set(str(result.get("message", "Ollama is unavailable. Start Ollama and try again.")))
        self.run_task("Checking local Ollama connection", client.status, done)

    def run_task(self, label, work, on_success):
        if self.busy:
            return
        self.busy = True
        self.status.set(label + "...")
        self.progress.start(12)
        alive = []
        for button in self.task_buttons:
            if button.winfo_exists():
                button.configure(state="disabled")
                alive.append(button)
        self.task_buttons = alive
        def run():
            try:
                self.work_queue.put((True, work(), on_success))
            except Exception as exc:
                self.work_queue.put((False, str(exc), None))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            success, result, callback = self.work_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.progress.stop()
            for button in self.task_buttons:
                if button.winfo_exists():
                    button.configure(state="normal")
            if success:
                self.status.set("Done. Ready for the next step.")
                try:
                    callback(result)
                except Exception as exc:
                    self.status.set("Could not update the view.")
                    messagebox.showerror("Display error", str(exc))
            else:
                self.status.set("Please check the message and try again.")
                messagebox.showerror("Action could not be completed", result)
        self.root.after(80, self._poll)

    def close(self):
        if self.busy:
            messagebox.showinfo("Operation in progress", "Please wait for the current operation to finish before closing. Local AI requests may take up to the configured timeout.")
            return
        self.root.destroy()
