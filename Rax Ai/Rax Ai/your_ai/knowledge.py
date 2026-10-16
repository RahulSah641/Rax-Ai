"""Persistent document knowledge base with honest offline and optional local AI modes.

All ordinary operations open their own SQLite connection. Writes are serialized,
transactions are atomic, and no connection is shared with a GUI worker thread.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import unicodedata
import uuid

from .ollama_client import OllamaError, validate_vector
from .sample_data import SAMPLE_DOCUMENTS
from .vectors import HNSW

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 200000
MAX_BACKUP_BYTES = 50 * 1024 * 1024
MAX_BACKUP_DOCUMENTS = 1000
MAX_BACKUP_CHARACTERS = 10000000
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35
BACKUP_FORMAT = "your-own-ai-backup"
BACKUP_VERSION = 1

STOPWORDS = frozenset("a an and are as at be been but by can could did do does for from had has have how "
                     "i if in into is it its me my of on or our should so than that the their them there these "
                     "they this those to was we were what when where which who why will with would you your".split())


class KnowledgeError(ValueError):
    """A readable validation or storage error for the user interface."""


def tokenize(text: str) -> list[str]:
    """Keep combining marks attached to letters, including Devanagari vowel signs."""
    terms, chars = [], []
    for char in unicodedata.normalize("NFC", text).casefold():
        if unicodedata.category(char)[0] in "LMN":
            chars.append(char)
        elif chars:
            term = "".join(chars)
            if term not in STOPWORDS:
                terms.append(term)
            chars = []
    if chars:
        term = "".join(chars)
        if term not in STOPWORDS:
            terms.append(term)
    return terms


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Slice original text spans; whitespace and punctuation remain unchanged."""
    if not isinstance(size, int) or not isinstance(overlap, int) or size < 1 or not 0 <= overlap < size:
        raise KnowledgeError("Chunk overlap must be smaller than a positive chunk size.")
    words = list(re.finditer(r"\S+", text))
    chunks = []
    for start in range(0, len(words), size - overlap):
        end = min(start + size, len(words))
        chunks.append(text[words[start].start():words[end - 1].end()])
        if end == len(words):
            break
    return chunks


def _validate_document(title: str, text: str) -> tuple[str, str]:
    if not isinstance(title, str) or not title.strip():
        raise KnowledgeError("Give the document a title.")
    title = title.strip()
    if len(title) > 240 or "\x00" in title:
        raise KnowledgeError("Document titles must be at most 240 characters and contain no null characters.")
    if not isinstance(text, str) or not text.strip():
        raise KnowledgeError("The document has no readable text. Add some text before saving.")
    if len(text) > MAX_TEXT_CHARS:
        raise KnowledgeError("Each document must contain at most 200,000 characters. Split this document into parts.")
    if "\x00" in text:
        raise KnowledgeError("The document contains null bytes and appears to be a binary file.")
    return title, text.strip()


def read_document(path: str | Path) -> str:
    """Read UTF-8 TXT/Markdown, or an optional text-based PDF (not scanned OCR)."""
    source = Path(path)
    try:
        if not source.is_file():
            raise KnowledgeError("Choose an existing document file.")
        if source.stat().st_size > MAX_FILE_BYTES:
            raise KnowledgeError("The file exceeds 10 MB. Split or reduce it before importing.")
        extension = source.suffix.lower()
        if extension in {".txt", ".md", ".markdown"}:
            with source.open("rb") as handle:
                raw = handle.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                raise KnowledgeError("The file exceeds 10 MB.")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise KnowledgeError("Save this text file with UTF-8 encoding, then import it again.") from exc
        elif extension == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise KnowledgeError("PDF import needs the optional pypdf library. Run: python -m pip install pypdf. "
                                     "You can import TXT and Markdown files without it.") from exc
            try:
                reader = PdfReader(str(source))
                if reader.is_encrypted and not reader.decrypt(""):
                    raise KnowledgeError("This PDF is password protected. Import an unlocked copy.")
                pages, count = [], 0
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    count += len(extracted) + 2
                    if count > MAX_TEXT_CHARS:
                        raise KnowledgeError("The PDF contains more than 200,000 text characters. Split it into parts.")
                    pages.append(extracted)
                text = "\n\n".join(pages)
                if not text.strip():
                    raise KnowledgeError("No selectable text was found in this PDF. Scanned PDFs need OCR first; "
                                         "this application does not perform OCR.")
            except KnowledgeError:
                raise
            except Exception as exc:
                raise KnowledgeError("Could not read this PDF. Check that it is a valid, unlocked text-based PDF.") from exc
        else:
            raise KnowledgeError("Supported files are UTF-8 .txt, .md, .markdown and text-based .pdf.")
    except OSError as exc:
        raise KnowledgeError(f"Could not read the file: {exc.strerror or str(exc)}") from exc
    return _validate_document(source.stem[:240] or "Document", text)[1]


def _fingerprint(title: str, text: str) -> str:
    return hashlib.sha256((title + "\0" + text).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeStore:
    def __init__(self, db_path: str | Path):
        self._lock = threading.RLock()
        self._closed = False
        self._keeper = None
        self._semantic_index = None
        self._semantic_signature = None
        self.db_path = str(db_path)
        self._uri = self.db_path == ":memory:"
        if self._uri:
            self.db_path = "file:your_ai_" + uuid.uuid4().hex + "?mode=memory&cache=shared"
            self._keeper = sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        else:
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(Path(self.db_path).expanduser())
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_key TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, model_key)
                );
            """)

    @contextmanager
    def _connect(self):
        if self._closed:
            raise KnowledgeError("The knowledge store is closed.")
        connection = sqlite3.connect(self.db_path, timeout=30, uri=self._uri)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _insert(connection, title: str, text: str, created_at: str | None = None) -> tuple[str, bool]:
        digest = _fingerprint(title, text)
        row = connection.execute("SELECT id FROM documents WHERE fingerprint=?", (digest,)).fetchone()
        if row:
            return row["id"], False
        document_id = uuid.uuid4().hex
        connection.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (document_id, title, text, created_at or _now(), digest))
        connection.executemany("INSERT INTO chunks VALUES (?,?,?,?)",
                               [(f"{document_id}:{i}", document_id, i, chunk)
                                for i, chunk in enumerate(chunk_text(text))])
        return document_id, True

    def add_document(self, title: str, text: str) -> str:
        title, text = _validate_document(title, text)
        with self._lock, self._connect() as connection:
            document_id, _ = self._insert(connection, title, text)
        return document_id

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""SELECT d.id, d.title, d.created_at,
                length(d.text) AS characters, count(c.id) AS chunks FROM documents d
                LEFT JOIN chunks c ON d.id=c.document_id GROUP BY d.id
                ORDER BY d.created_at DESC, d.title COLLATE NOCASE, d.id""").fetchall()
        return [dict(row) for row in rows]

    def get_document(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("""SELECT d.id,d.title,d.text,d.created_at,length(d.text) AS characters,
                (SELECT count(*) FROM chunks WHERE document_id=d.id) AS chunks
                FROM documents d WHERE id=?""", (document_id,)).fetchone()
        return dict(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        with self._lock, self._connect() as connection:
            result = connection.execute("DELETE FROM documents WHERE id=?", (document_id,))
            return result.rowcount > 0

    def stats(self) -> dict:
        with self._connect() as connection:
            documents, characters = connection.execute("SELECT count(*),coalesce(sum(length(text)),0) FROM documents").fetchone()
            chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            embeddings = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        return {"documents": documents, "chunks": chunks, "characters": characters, "cached_embeddings": embeddings}

    def _chunks(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("""SELECT c.id AS chunk_id,c.document_id,
                d.title,c.text FROM chunks c JOIN documents d ON c.document_id=d.id
                ORDER BY c.document_id,c.ordinal""").fetchall()]

    @staticmethod
    def _validate_query(query: str, k: int):
        if not isinstance(query, str) or not query.strip():
            raise KnowledgeError("Enter a question or a few search words.")
        if len(query) > 4000:
            raise KnowledgeError("Search questions must be at most 4,000 characters.")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
            raise KnowledgeError("The number of search results must be between 1 and 20.")

    def search(self, query: str, k: int = 5, mode: str = "offline", client=None) -> list[dict]:
        self._validate_query(query, k)
        if mode not in {"offline", "ollama", "semantic"}:
            raise KnowledgeError("Choose Offline or Local AI (Ollama) mode.")
        chunks = self._chunks()
        if not chunks:
            return []
        if mode != "offline":
            if client is None:
                raise KnowledgeError("Local AI search needs an Ollama client. Configure Ollama or use Offline mode.")
            return self._semantic_search(query, chunks, k, client)
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return []
        counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        document_frequency = Counter(term for count in counts for term in count)
        idf = {term: math.log((len(chunks) + 1) / (frequency + 1)) + 1
               for term, frequency in document_frequency.items()}
        query_weights = {term: (1 + math.log(count)) * idf[term]
                         for term, count in query_counts.items() if term in idf}
        query_norm = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
        if not query_norm:
            return []
        scored = []
        for chunk, count in zip(chunks, counts):
            if not query_weights.keys() & count.keys():
                continue
            weights = {term: (1 + math.log(amount)) * idf[term] for term, amount in count.items()}
            norm = math.sqrt(sum(value ** 2 for value in weights.values()))
            score = sum(weights.get(term, 0) * value for term, value in query_weights.items()) / (norm * query_norm)
            if score > 0:
                scored.append({**chunk, "score": score})
        return sorted(scored, key=lambda row: (-row["score"], row["chunk_id"]))[:k]

    def _semantic_search(self, query: str, chunks: list[dict], k: int, client) -> list[dict]:
        # The cache key includes endpoint, model name and model-content digest.
        key = client.embedding_cache_key()
        vectors = {}
        with self._connect() as connection:
            cached = {row["chunk_id"]: row for row in connection.execute(
                "SELECT * FROM embeddings WHERE model_key=?", (key,)).fetchall()}
        missing = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
            row = cached.get(chunk_id)
            if row and row["text_hash"] == text_hash:
                try:
                    vectors[chunk_id] = validate_vector(json.loads(row["vector_json"]))
                    continue
                except (ValueError, TypeError):
                    pass
            missing.append((chunk, text_hash))
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            result = client.embed([chunk["text"] for chunk, _ in batch])
            if len(result) != len(batch):
                raise OllamaError("The embedding service returned the wrong number of vectors.")
            writes = []
            for (chunk, text_hash), vector in zip(batch, result):
                vector = validate_vector(vector)
                vectors[chunk["chunk_id"]] = vector
                writes.append((chunk["chunk_id"], key, text_hash, json.dumps(vector, allow_nan=False)))
            with self._lock, self._connect() as connection:
                # A document may have been deleted while an HTTP request was in flight.
                for values in writes:
                    connection.execute("""INSERT OR REPLACE INTO embeddings
                        SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM chunks WHERE id=?)""", values + (values[0],))
        query_vectors = client.embed([query])
        if len(query_vectors) != 1:
            raise OllamaError("Ollama returned an invalid query embedding.")
        query_vector = validate_vector(query_vectors[0])
        if any(len(v) != len(query_vector) for v in vectors.values()):
            # Discard stale vectors, never compare vectors with incompatible dimensions.
            with self._lock, self._connect() as connection:
                connection.execute("DELETE FROM embeddings WHERE model_key=?", (key,))
                self._semantic_index = None
                self._semantic_signature = None
            raise OllamaError("Embedding dimensions changed. Cached vectors were cleared; retry the search.")
        # Cache the in-memory HNSW graph until document membership or the model
        # content digest changes. Database vectors remain the durable source.
        signature = (key, tuple(chunk["chunk_id"] for chunk in chunks))
        with self._lock:
            if self._semantic_signature != signature:
                index = HNSW({i: vectors[chunk["chunk_id"]] for i, chunk in enumerate(chunks)},
                             m=8, ef_construction=64, ef_search=max(32, k), seed=42)
                index.prepare("cosine")
                self._semantic_index = index
                self._semantic_signature = signature
            hits = self._semantic_index.search(query_vector, k=min(k, len(chunks)), metric="cosine")
        return [{**chunks[index], "score": max(-1.0, min(1.0, 1.0 - distance))}
                for index, distance in hits]

    def ask(self, question: str, k: int = 4, mode: str = "offline", client=None) -> dict:
        if mode != "offline" and k > 8:
            raise KnowledgeError("Use at most 8 source passages for Local AI answers.")
        sources = self.search(question, k=k, mode=mode, client=client)
        effective_mode = "offline" if mode == "offline" else "ollama"
        if not sources:
            return {"answer": "No matching evidence was found in your documents. Add a relevant document or "
                    "try the exact keywords used in it.", "sources": [], "mode": effective_mode}
        if mode != "offline":
            return {"answer": client.generate(question, sources), "sources": sources, "mode": "ollama"}
        # Return only copied sentences, with source references. This is extraction,
        # not a generated answer, and keyword overlap is not proof of relevance.
        terms = set(tokenize(question))
        answer_lines = ["Offline mode — matching source excerpts (not a generated answer):"]
        for number, source in enumerate(sources, 1):
            sentences = [part.strip() for part in re.split(r"(?<=[.!?।])\s+|\n\s*\n", source["text"]) if part.strip()]
            ranked = sorted(enumerate(sentences), key=lambda item: (-len(terms & set(tokenize(item[1]))), item[0]))
            excerpt = ranked[0][1] if ranked else source["text"]
            if len(excerpt) > 900:
                excerpt = excerpt[:900].rsplit(" ", 1)[0] + "…"
            answer_lines.append(f"[{number}] {excerpt}")
        answer_lines.append("Open the cited passages to check the context. Keyword matches may not fully answer the question.")
        return {"answer": "\n\n".join(answer_lines), "sources": sources, "mode": "offline"}

    def export_backup(self, path: str | Path):
        with self._connect() as connection:
            documents = [dict(row) for row in connection.execute(
                "SELECT title,text,created_at FROM documents ORDER BY created_at,title,id").fetchall()]
        archive = {"format": BACKUP_FORMAT, "version": BACKUP_VERSION, "exported_at": _now(), "documents": documents}
        if len(documents) > MAX_BACKUP_DOCUMENTS or sum(len(d["text"]) for d in documents) > MAX_BACKUP_CHARACTERS:
            raise KnowledgeError("The knowledge base exceeds the portable backup limit (1,000 documents or 10 million characters).")
        raw = json.dumps(archive, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        if len(raw) > MAX_BACKUP_BYTES:
            raise KnowledgeError("This backup exceeds the 50 MB portable backup limit.")
        destination = Path(path)
        if not self._uri and destination.resolve() == Path(self.db_path).resolve():
            raise KnowledgeError("Choose a JSON backup file, not the active SQLite database.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, prefix=".backup-", suffix=".tmp", delete=False) as handle:
                temp_path = handle.name
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def import_backup(self, path: str | Path) -> int:
        try:
            with Path(path).open("rb") as handle:
                raw = handle.read(MAX_BACKUP_BYTES + 1)
            if len(raw) > MAX_BACKUP_BYTES:
                raise KnowledgeError("This backup exceeds the 50 MB import limit.")
            data = json.loads(raw.decode("utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            if isinstance(exc, KnowledgeError):
                raise
            raise KnowledgeError("Could not read this backup. Choose a valid Rax Ai JSON backup.") from exc
        if not isinstance(data, dict) or data.get("format") != BACKUP_FORMAT or type(data.get("version")) is not int or data.get("version") != BACKUP_VERSION:
            raise KnowledgeError("Unsupported backup format or version. Import a Rax Ai version 1 backup.")
        rows = data.get("documents")
        if not isinstance(rows, list) or len(rows) > MAX_BACKUP_DOCUMENTS:
            raise KnowledgeError("Backup documents must be a list containing at most 1,000 items.")
        validated, characters = [], 0
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise KnowledgeError(f"Backup document {index} is invalid. Nothing was imported.")
            try:
                title, text = _validate_document(row.get("title"), row.get("text"))
            except KnowledgeError as exc:
                raise KnowledgeError(f"Backup document {index}: {exc} Nothing was imported.") from exc
            characters += len(text)
            if characters > MAX_BACKUP_CHARACTERS:
                raise KnowledgeError("The backup exceeds 10 million text characters. Nothing was imported.")
            created_at = row.get("created_at")
            if created_at is not None:
                try:
                    if not isinstance(created_at, str) or len(created_at) > 40:
                        raise ValueError()
                    datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise KnowledgeError(f"Backup document {index} has an invalid creation date. Nothing was imported.") from exc
            validated.append((title, text, created_at))
        # Validation finishes before the transaction; only raw document text is
        # imported. IDs, chunks and embeddings from an untrusted backup are ignored.
        imported = 0
        with self._lock, self._connect() as connection:
            for title, text, created_at in validated:
                _, added = self._insert(connection, title, text, created_at)
                imported += int(added)
        return imported

    def load_samples(self) -> int:
        with self._lock, self._connect() as connection:
            return sum(int(self._insert(connection, sample["title"], sample["text"])[1]) for sample in SAMPLE_DOCUMENTS)

    def close(self):
        with self._lock:
            if self._keeper is not None:
                self._keeper.close()
                self._keeper = None
            self._closed = True
