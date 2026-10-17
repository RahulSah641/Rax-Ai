"""Headless controller integration checks; these do not validate Tk rendering.

Only widgets and dialogs are stand-ins. SQLite, retrieval, indexes, worker
threads and completion callbacks all use the real application code.
"""
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    from your_ai.gui import Application
except ImportError:  # Backend-only installations may omit the Tk module.
    Application = None

from your_ai.knowledge import KnowledgeStore
from your_ai.settings import DEFAULTS
from your_ai.vectors import VectorLab


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Text:
    def __init__(self):
        self.text = ""

    def configure(self, **kwargs):
        pass

    def delete(self, start, end):
        self.text = ""

    def insert(self, index, text):
        self.text = text


class Tree:
    def __init__(self):
        self.rows = {}
        self.selected = ()
        self.next_id = 0

    def selection(self):
        return self.selected

    def selection_set(self, item):
        if item not in self.rows:
            raise ValueError("Cannot select an absent tree row")
        self.selected = (item,)

    def get_children(self):
        return tuple(self.rows)

    def exists(self, item):
        return item in self.rows

    def delete(self, *items):
        for item in items:
            self.rows.pop(item, None)
        self.selected = tuple(item for item in self.selected if item in self.rows)

    def insert(self, parent, position, iid=None, values=()):
        if iid is None:
            iid = str(self.next_id)
            self.next_id += 1
        if iid in self.rows:
            raise ValueError("Duplicate tree row")
        self.rows[iid] = values

    def item(self, item, option):
        return self.rows[item]


@unittest.skipIf(Application is None, "Tkinter module is not installed")
class GUIControllerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = KnowledgeStore(Path(self.folder.name) / "knowledge.sqlite3")
        self.addCleanup(self.store.close)
        self.app = Application.__new__(Application)
        app = self.app
        app.root = Mock()
        app.store = self.store
        app.lab = VectorLab()
        app.settings = DEFAULTS.copy()
        app.work_queue = queue.Queue()
        app.busy = False
        app.task_buttons = [Mock()]
        app.status = Value()
        app.progress = Mock()
        app.doc_tree = Tree()
        app.doc_title = Value()
        app.doc_text = Text()
        app.doc_count = Value()
        app.stat_vars = {name: Value() for name in ("documents", "chunks", "vectors")}
        app.question = Value()
        app.ask_mode = Value("Offline (exact extracts)")
        app.sources = []
        app.last_answer = ""
        app.answer_label = Value()
        app.answer_text = Text()
        app.source_tree = Tree()
        app.source_text = Text()
        app.vector_query = Value(app.lab.items[0]["label"])
        app.algorithm = Value("kd_tree")
        app.metric = Value("cosine")
        app.top_k = Value("5")
        app.result_tree = Tree()
        app.result_label = Value()
        app.benchmark_tree = Tree()
        app.graph_note = Value()
        app.highlight_ids = set()
        error_patch = patch("your_ai.gui.messagebox.showerror")
        self.error = error_patch.start()
        self.addCleanup(error_patch.stop)

    def finish_task(self):
        # Wait for a real background operation, then process its GUI completion.
        result = self.app.work_queue.get(timeout=5)
        self.app.work_queue.put(result)
        self.app._poll()
        self.assertFalse(self.app.busy)
        self.app.task_buttons[0].configure.assert_called_with(state="normal")

    def test_select_persisted_document_with_uuid_id(self):
        document_id = self.store.add_document("SQLite notes", "SQLite saves documents on your computer.")
        self.app.refresh_documents()
        self.app.doc_tree.selection_set(document_id)
        self.app.select_document()
        self.assertEqual(self.app.doc_title.get(), "SQLite notes")
        self.assertEqual(self.app.doc_text.text, "SQLite saves documents on your computer.")

    def test_stale_selection_after_another_process_deletes_document(self):
        document_id = self.store.add_document("Temporary", "Some temporary document text.")
        self.app.refresh_documents()
        self.app.doc_tree.selection_set(document_id)
        self.store.delete_document(document_id)
        self.app.select_document()
        self.assertEqual(self.app.doc_tree.get_children(), ())
        self.assertIn("no longer available", self.app.doc_text.text)

    def test_delete_selected_document_updates_storage_and_view(self):
        document_id = self.store.add_document("Remove me", "There is one searchable document.")
        self.app.refresh_documents()
        self.app.doc_tree.selection_set(document_id)
        with patch("your_ai.gui.messagebox.askyesno", return_value=True):
            self.app.delete_document()
        self.finish_task()
        self.assertIsNone(self.store.get_document(document_id))
        self.assertEqual(self.store.stats()["chunks"], 0)
        self.assertEqual(self.app.doc_tree.get_children(), ())
        self.assertIn("deleted", self.app.doc_text.text)
        self.error.assert_not_called()

    def test_offline_answer_renders_actual_evidence_and_sources(self):
        text = "An SQLite database stores project documents locally on the computer."
        self.store.add_document("Project storage", text)
        self.app.question.set("Where does SQLite store documents?")
        self.app.ask_question()
        self.finish_task()
        self.assertIn(text, self.app.answer_text.text)
        self.assertIn(text, self.app.source_text.text)
        self.assertEqual(self.app.source_tree.rows["0"][1], "Project storage")
        self.assertIn("[1] Project storage", self.app.last_answer)
        self.error.assert_not_called()

    def test_invalid_saved_ollama_settings_have_visible_error(self):
        self.app.settings["ollama_url"] = "https://example.com"
        self.app.question.set("Explain my notes")
        self.app.ask_mode.set("Ollama (local AI)")
        self.app.ask_question()
        self.assertFalse(self.app.busy)
        self.assertIn("Settings", self.error.call_args.args[1])

    def test_vector_query_search_and_benchmark_render_measured_results(self):
        self.app.search_vectors()
        self.finish_task()
        rows = list(self.app.result_tree.rows.values())
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0][0], self.app.lab.items[0]["label"])
        self.assertAlmostEqual(float(rows[0][2]), 0)
        self.app.compare_vectors()
        self.finish_task()
        rows = {row[0]: row for row in self.app.benchmark_tree.rows.values()}
        self.assertEqual(set(rows), {"brute_force", "kd_tree", "hnsw"})
        self.assertEqual(rows["kd_tree"][2], "100%")
        self.assertTrue(all(float(row[1]) >= 0 for row in rows.values()))
        self.error.assert_not_called()

    def test_backend_failure_releases_busy_state_and_shows_message(self):
        self.app.vector_query.set("[1, 2]")
        self.app.search_vectors()
        self.finish_task()
        self.error.assert_called_once()
        self.assertIn("16", self.error.call_args.args[1])
        self.assertEqual(self.app.result_tree.get_children(), ())


if __name__ == "__main__":
    unittest.main()
