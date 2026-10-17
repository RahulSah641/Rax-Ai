import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from your_ai.knowledge import (BACKUP_FORMAT, CHUNK_OVERLAP, CHUNK_WORDS, KnowledgeError,
                               KnowledgeStore, chunk_text, read_document, tokenize)
from your_ai.ollama_client import OllamaError


class FakeOllama:
    def __init__(self):
        self.key = "model|digest-v1"
        self.embedded = []
        self.generation_calls = 0

    def embedding_cache_key(self):
        return self.key

    def embed(self, texts):
        self.embedded.extend(texts)
        return [[float("solar" in text.lower()) + .1, float("python" in text.lower()) + .1, .1] for text in texts]

    def generate(self, question, sources):
        self.generation_calls += 1
        return "Generated test answer [1]"


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.db_path = self.path / "knowledge.sqlite3"
        self.store = KnowledgeStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def write_backup(self, documents, **extra):
        path = self.path / "backup.json"
        data = {"format": BACKUP_FORMAT, "version": 1, "documents": documents, **extra}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def test_crud_persistence_duplicate_and_cascade(self):
        ident = self.store.add_document("Notes", "Solar panels convert sunlight into electricity.")
        self.assertEqual(ident, self.store.add_document("Notes", "Solar panels convert sunlight into electricity."))
        self.assertEqual(self.store.stats()["documents"], 1)
        second = KnowledgeStore(self.db_path)
        self.assertEqual(second.get_document(ident)["title"], "Notes")
        self.assertEqual(second.list_documents()[0]["chunks"], 1)
        second.close()
        self.store.search("solar", mode="ollama", client=FakeOllama())
        self.assertEqual(self.store.stats()["cached_embeddings"], 1)
        self.assertTrue(self.store.delete_document(ident))
        self.assertFalse(self.store.delete_document(ident))
        self.assertIsNone(self.store.get_document(ident))
        self.assertEqual(self.store.stats(), {"documents": 0, "chunks": 0, "characters": 0, "cached_embeddings": 0})

    def test_in_memory_store_works_with_per_operation_connections(self):
        memory = KnowledgeStore(":memory:")
        memory.add_document("Example", "Testing persistence.")
        self.assertEqual(memory.stats()["documents"], 1)
        memory.close()
        with self.assertRaises(KnowledgeError):
            memory.stats()

    def test_chunk_overlap_and_exact_original_spans(self):
        text = "\n  ".join("word" + str(i) for i in range(420))
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), CHUNK_WORDS)
        self.assertEqual(chunks[0].split()[-CHUNK_OVERLAP:], chunks[1].split()[:CHUNK_OVERLAP])
        self.assertEqual(chunks[-1].split()[-1], "word419")
        self.assertTrue(all(chunk in text for chunk in chunks))
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("single"), ["single"])
        with self.assertRaises(KnowledgeError):
            chunk_text("bad", size=10, overlap=10)

    def test_unicode_tokenization_and_search(self):
        self.store.add_document("मराठी", "ज्ञान हेच खरे धन आहे. मराठी पुस्तके वाचा.")
        self.assertIn("पुस्तके", tokenize("मराठी पुस्तके!"))
        rows = self.store.search("पुस्तके")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "मराठी")

    def test_offline_ranking_and_no_match(self):
        self.store.add_document("Solar", "Solar panels create electricity from sunlight.")
        self.store.add_document("Programming", "Python provides database and desktop tools.")
        self.assertEqual(self.store.search("solar sunlight")[0]["title"], "Solar")
        self.assertEqual(self.store.search("penguins antarctica"), [])
        self.assertEqual(self.store.search("the and of"), [])
        answer = self.store.ask("penguins")
        self.assertEqual(answer["sources"], [])
        self.assertIn("No matching evidence", answer["answer"])

    def test_offline_answer_has_exact_cited_evidence(self):
        passage = "Plants grow in soil. Solar panels convert sunlight into electricity. Batteries store energy."
        self.store.add_document("Energy", passage)
        answer = self.store.ask("How do solar panels work?")
        self.assertEqual(answer["mode"], "offline")
        self.assertIn("[1] Solar panels convert sunlight into electricity.", answer["answer"])
        self.assertIn("not a generated answer", answer["answer"])
        self.assertEqual(answer["sources"][0]["text"], passage)

    def test_validation(self):
        for title, text in [("", "text"), ("a" * 241, "text"), ("title", "  "), ("title", "x" * 200001), ("title", "bin\x00ary")]:
            with self.subTest(title=title[:10]), self.assertRaises(KnowledgeError):
                self.store.add_document(title, text)
        for query, k in [("", 5), ("solar", 0), ("solar", True), ("solar", 21), ("x" * 4001, 5)]:
            with self.assertRaises(KnowledgeError):
                self.store.search(query, k=k)
        with self.assertRaises(KnowledgeError):
            self.store.search("solar", mode="magic")

    def test_samples_idempotent_and_answerable(self):
        self.assertEqual(self.store.load_samples(), 4)
        self.assertEqual(self.store.load_samples(), 0)
        self.assertTrue(self.store.ask("How does offline mode work?")["sources"])

    def test_backup_roundtrip_merges_and_ignores_derived_data(self):
        ident = self.store.add_document("मराठी", "ज्ञान हेच खरे धन आहे.")
        path = self.path / "export.json"
        self.store.export_backup(path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("embeddings", exported)
        exported["documents"][0].update({"id": "attacker-id", "chunks": ["poisoned"], "embeddings": [[99]]})
        path.write_text(json.dumps(exported), encoding="utf-8")
        target = KnowledgeStore(self.path / "other.sqlite3")
        self.assertEqual(target.import_backup(path), 1)
        self.assertEqual(target.import_backup(path), 0)
        document = target.list_documents()[0]
        self.assertNotEqual(document["id"], ident)
        self.assertNotEqual(document["id"], "attacker-id")
        self.assertEqual(target.stats()["cached_embeddings"], 0)
        self.assertEqual(target.search("ज्ञान")[0]["text"], "ज्ञान हेच खरे धन आहे.")
        target.close()

    def test_backup_invalid_document_changes_nothing(self):
        self.store.add_document("Original", "Keep this document.")
        path = self.write_backup([{"title": "Valid", "text": "Valid new document."}, {"title": "Invalid", "text": " "}])
        with self.assertRaisesRegex(KnowledgeError, "Nothing was imported"):
            self.store.import_backup(path)
        self.assertEqual([row["title"] for row in self.store.list_documents()], ["Original"])

    def test_backup_database_failure_rolls_back(self):
        path = self.write_backup([{"title": "First", "text": "first document"}, {"title": "Second", "text": "second document"}])
        original = self.store._insert
        calls = [0]
        def fail_second(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise sqlite3.OperationalError("simulated disk failure")
            return original(*args, **kwargs)
        with patch.object(self.store, "_insert", side_effect=fail_second), self.assertRaises(sqlite3.OperationalError):
            self.store.import_backup(path)
        self.assertEqual(self.store.stats()["documents"], 0)
        self.assertEqual(self.store.stats()["chunks"], 0)

    def test_backup_schema_and_date_rejections(self):
        bad = [{"format": "foreign", "version": 1, "documents": []},
               {"format": BACKUP_FORMAT, "version": True, "documents": []},
               {"format": BACKUP_FORMAT, "version": 1, "documents": {}},
               {"format": BACKUP_FORMAT, "version": 1, "documents": [{"title": "A", "text": "B", "created_at": "not-a-date"}]}]
        path = self.path / "bad.json"
        for data in bad:
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                self.store.import_backup(path)
        path.write_text("[broken", encoding="utf-8")
        with self.assertRaises(KnowledgeError):
            self.store.import_backup(path)

    def test_export_atomic_failure_preserves_previous_file(self):
        path = self.path / "existing.json"
        path.write_text("keep this", encoding="utf-8")
        with patch("your_ai.knowledge.os.replace", side_effect=OSError("blocked")), self.assertRaises(OSError):
            self.store.export_backup(path)
        self.assertEqual(path.read_text(encoding="utf-8"), "keep this")
        self.assertEqual(list(self.path.glob(".backup-*")), [])

    def test_backup_cannot_overwrite_live_database(self):
        self.store.add_document("Keep", "Important content.")
        with self.assertRaisesRegex(KnowledgeError, "active SQLite"):
            self.store.export_backup(self.db_path)
        self.assertEqual(self.store.stats()["documents"], 1)

    def test_parallel_writes_use_separate_connections(self):
        errors = []
        def add(index):
            try:
                self.store.add_document(f"Thread {index}", f"Document number {index}")
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=add, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
        self.assertEqual(self.store.stats()["documents"], 12)

    def test_semantic_cache_model_invalidation_and_answer(self):
        self.store.add_document("Solar", "Solar panels generate electricity.")
        self.store.add_document("Python", "Python is a programming language.")
        client = FakeOllama()
        result = self.store.search("solar", mode="ollama", client=client)
        self.assertEqual(result[0]["title"], "Solar")
        self.assertEqual(len(client.embedded), 3)
        self.store.search("solar", mode="ollama", client=client)
        self.assertEqual(len(client.embedded), 4)  # only query embedding is new
        client.key = "model|digest-v2"
        self.store.search("python", mode="ollama", client=client)
        self.assertEqual(len(client.embedded), 7)
        answer = self.store.ask("solar", mode="ollama", client=client)
        self.assertEqual(answer["mode"], "ollama")
        self.assertEqual(answer["answer"], "Generated test answer [1]")
        self.assertEqual(client.generation_calls, 1)

    def test_semantic_failure_is_not_disguised_as_offline(self):
        self.store.add_document("A", "Solar electricity.")
        with self.assertRaises(KnowledgeError):
            self.store.search("solar", mode="ollama")
        client = FakeOllama()
        with patch.object(client, "embed", side_effect=OllamaError("model missing")), self.assertRaisesRegex(OllamaError, "model missing"):
            self.store.ask("solar", mode="ollama", client=client)

    def test_invalid_cached_vector_is_recomputed(self):
        self.store.add_document("Solar", "Solar electricity.")
        client = FakeOllama()
        self.store.search("solar", mode="ollama", client=client)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("UPDATE embeddings SET vector_json='[0,0,0]'")
        self.store.search("solar", mode="ollama", client=client)
        self.assertEqual(len(client.embedded), 4)

    def test_semantic_dimension_change_clears_cache(self):
        self.store.add_document("Solar", "Solar electricity.")
        client = FakeOllama()
        self.store.search("solar", mode="ollama", client=client)
        with patch.object(client, "embed", return_value=[[1.0, .1]]), self.assertRaisesRegex(OllamaError, "dimensions changed"):
            self.store.search("solar", mode="ollama", client=client)
        self.assertEqual(self.store.stats()["cached_embeddings"], 0)
        with patch.object(client, "embed", side_effect=lambda texts: [[1.0, .1] for _ in texts]):
            self.assertTrue(self.store.search("solar", mode="ollama", client=client))

    def test_text_file_import_utf8_and_bounds(self):
        path = self.path / "मराठी.md"
        path.write_text("\ufeffमराठी पुस्तके", encoding="utf-8")
        self.assertEqual(read_document(path), "मराठी पुस्तके")
        path.write_bytes(b"\xff\xfe\xfa")
        with self.assertRaisesRegex(KnowledgeError, "UTF-8"):
            read_document(path)
        path.write_bytes(b"binary\x00data")
        with self.assertRaisesRegex(KnowledgeError, "binary"):
            read_document(path)
        with path.open("wb") as handle:
            handle.truncate(10 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(KnowledgeError, "10 MB"):
            read_document(path)
        unknown = self.path / "word.docx"
        unknown.write_text("data")
        with self.assertRaisesRegex(KnowledgeError, "Supported"):
            read_document(unknown)

    def test_optional_pdf_error_and_text_extraction(self):
        path = self.path / "notes.pdf"
        path.write_bytes(b"%PDF-placeholder-for-mocked-reader")
        with patch.dict(sys.modules, {"pypdf": None}), self.assertRaisesRegex(KnowledgeError, "pip install pypdf"):
            read_document(path)
        reader = types.SimpleNamespace(is_encrypted=False, pages=[types.SimpleNamespace(extract_text=lambda: "PDF sample text")])
        fake = types.SimpleNamespace(PdfReader=lambda _: reader)
        with patch.dict(sys.modules, {"pypdf": fake}):
            self.assertEqual(read_document(path), "PDF sample text")
        reader.pages = [types.SimpleNamespace(extract_text=lambda: "")]
        with patch.dict(sys.modules, {"pypdf": fake}), self.assertRaisesRegex(KnowledgeError, "OCR"):
            read_document(path)


if __name__ == "__main__":
    unittest.main()
