# Rax Ai - Python Desktop Edition

## 1. Executive overview

Rax Ai is an educational desktop application that brings vector search, document retrieval and optional local generative AI into one Python project. It reimplements the central ideas in the supplied C++ archive and adds an accessible desktop interface, local persistence, an offline document-search path and reproducible validation.

The primary objective is practical understanding. A learner should be able to start the project, inspect real search results, compare algorithms, import personal notes, ask questions and read the evidence behind an answer. The application is also a foundation for a personal document assistant or an information-retrieval assignment. It is not a trained foundation model, a hosted chat service or a production vector-database replacement.

The delivered package includes readable Python source, launch scripts, sample content, automated tests, this report and a presentation PDF. Optional PDF extraction and local neural inference are separate capabilities with their own requirements.

## 2. Problem statement and goals

The original project combines C++ vector algorithms with a browser interface. That is a useful systems-programming example, but compiler setup and mandatory model installation make the first experience harder for a beginner. The Python version reduces the setup burden and separates concepts that can otherwise be confused: synthetic demo vectors, lexical retrieval and neural retrieval-augmented generation.

The goals are to:

1. Preserve three distinct algorithms and three distance choices in understandable Python.
2. Make the first run useful without downloaded models or paid API keys.
3. Provide a simple desktop workflow with clear actions and source evidence.
4. Keep document content available after the application closes.
5. Measure search behavior honestly against an exact baseline.
6. Support optional local embeddings and generated answers when Ollama is installed.
7. Provide enough documentation and tests to explain, demonstrate and extend the project.

## 3. Delivered scope

### 3.1 Vector search laboratory

The demo uses a small collection of synthetic 16-dimensional vectors arranged around named subject categories. These vectors are intentionally constructed to make neighbors and clusters understandable. They are not outputs of a neural language model, and the label lookup should not be presented as general-purpose semantic search.

The user selects a concept or vector, a distance metric, the desired number of neighbors and an algorithm. The result displays nearest items and distance values. A PCA projection plots the dataset in two dimensions. The projection is visual context only: ranking uses the full vectors.

### 3.2 Document workspace

Users can paste text or import supported text files. Documents are divided into overlapping chunks and saved with their titles and source information in SQLite. TXT and Markdown work with the standard library. Text-based PDF import requires the optional `pypdf` dependency. The app does not read scans using OCR.

### 3.3 Offline source extraction

Offline retrieval builds term-based TF-IDF vectors from saved chunks and the question. It ranks relevant passages and returns source-labelled extracts. These extracts are existing text from the documents; they are not a newly composed language-model answer. Queries with no useful lexical overlap may find little or no evidence even if a human would recognize a synonym.

### 3.4 Optional local AI

Ollama provides a separate local inference service. The embedding model converts document chunks and questions into compatible dense vectors. Retrieved context is passed to a generation model to produce an answer. The application shows source context so the user can check what was retrieved. Retrieval improves grounding but does not guarantee factual accuracy or prevent fabricated claims.

### 3.5 Programmatic access

The local API exposes application actions for scripts and demonstrations. It binds to the loopback interface by default. It is intended for trusted, local development, with a bearer token for its API routes and no public deployment or multi-user access-control claim.

## 4. Technology stack and dependencies

| Component | Technology | Why it is used |
|---|---|---|
| Application language | Python 3.10+ | Readable classes, portable standard library and low setup effort |
| Recommended baseline | Python 3.12 | A practical interpreter baseline for this source package |
| Desktop interface | `tkinter`, `ttk` | Native desktop windows, tabs, tables and dialogs |
| Persistence | `sqlite3` | A local file database without a separate database server |
| Algorithms | Python `math`, `heapq`, `random` and collections | Transparent distance, graph, tree and ranking logic |
| File and configuration handling | `pathlib`, `json` and standard modules | Predictable paths, JSON settings and import/export |
| HTTP integration | Python standard HTTP client/server modules | Local Ollama requests and a loopback API |
| Automated testing | `unittest` | Tests that run without installing a test framework |
| Optional PDF input | `pypdf` | Extract text from PDFs with a text layer |
| Optional embeddings | Ollama + `nomic-embed-text` | Dense model-generated text vectors |
| Optional generation | Ollama + `llama3.2:1b` | Local text generation from retrieved context |
| Presentation regeneration | `reportlab` | Generates the bundled PDF; not required to run the app |

A standard Python installation needs Tk support for the GUI. Tkinter is not guaranteed in every Linux or minimal Python distribution. Test it with `python -m tkinter`. The core application has no third-party pip requirement. This does not mean it has no dependencies: it still needs Python, Tcl/Tk and the operating-system desktop facilities.

The Python source implements the project logic. Standard runtimes such as Tk and SQLite may contain native code internally; the user does not need a C++ compiler for this project.

## 5. Architecture and separation of concerns

The entry point chooses desktop or API mode. Both interfaces call the same application services, which coordinate the vector engine, document store and optional Ollama client. This keeps algorithm logic separate from interface code and makes core functions testable without opening a window.

There are three data paths:

1. **Demo path:** synthetic vector -> selected search algorithm -> ranked neighbors -> PCA display.
2. **Offline document path:** saved chunks -> TF-IDF representation -> relevance ranking -> labelled extracts.
3. **Ollama document path:** chunks/question -> model embeddings -> nearest chunks -> local generation -> answer plus retrieved sources.

SQLite keeps durable application data. Search indexes are working structures that can be recreated from stored content. A failed model request should appear as a visible error rather than as an invented answer.

## 6. Vector representation and distance metrics

A vector is an ordered list of numbers. Every vector in one index must have the same dimensionality. Comparing incompatible dimensions is invalid and must raise an error instead of silently discarding coordinates.

For vectors x and y with d coordinates:

- **Euclidean distance:** `sqrt(sum((x_i - y_i)^2))`. Measures straight-line separation. Large coordinates can dominate if features have different scales.
- **Manhattan distance:** `sum(abs(x_i - y_i))`. Adds coordinate-wise differences.
- **Cosine distance:** `1 - dot(x, y) / (norm(x) * norm(y))`. Compares directions. Similarity and distance are different displays: higher similarity is better; lower distance is better.

The app handles zero vectors explicitly because the usual cosine expression divides by their norm. Duplicate vectors and tied distances can produce multiple equally valid nearest neighbors; deterministic ordering helps make examples repeatable.

The 16-dimensional demo space and a document embedding space are separate. A model change can change dimensions or vector semantics. Documents must be re-embedded with the same model used for their questions. Synthetic vectors must never be mixed into a neural embedding index.

## 7. Search algorithms

### 7.1 Brute force

Brute force computes the query's distance to every stored vector and chooses the nearest results. It provides the exact reference used for recall measurement. Its distance computation costs O(N x d), where N is the number of vectors and d is the dimension. Selecting or sorting neighbors adds ranking work, depending on the implementation.

For a small collection, brute force can be the fastest practical choice because it has little index overhead. It is simple, reliable and important as an oracle for testing other algorithms.

### 7.2 KD-Tree

A KD-Tree repeatedly splits points along coordinate axes. A nearest-neighbor query visits the promising branch first, then checks whether another branch could contain a better result. Correct distance bounds allow unnecessary branches to be skipped.

KD-Tree is exact when the bounds and traversal are correct. Its speed depends on dimensions and data geometry. In high dimensions, pruning may become weak and search may approach a full scan. A blanket O(log N) query guarantee would be misleading. Cosine handling requires normalization or safe fallback behavior rather than applying an invalid axis-bound formula directly to cosine distance.

### 7.3 HNSW

Hierarchical Navigable Small World search uses a graph with multiple levels. Higher levels provide coarse navigation, while the lowest level contains the most detailed neighborhood connections. A query begins at the upper levels and refines its candidate set nearer the bottom.

HNSW is approximate. Parameters controlling graph connectivity, construction effort and search breadth affect memory, build time, query time and recall. A larger search effort can improve recall at a cost. No setting makes every dataset or runtime predictable. This project contains an educational Python implementation; production HNSW systems add substantial optimization and operational engineering.

### 7.4 Selecting an algorithm

Use brute force to establish truth for small experiments. Use KD-Tree to study exact spatial pruning and its dimensional limits. Use HNSW to explore an approximate graph method and its recall/latency trade-off. The correct choice follows measurements on representative data, not a fixed ranking of algorithm names.

## 8. PCA visualization

Principal component analysis centers the vectors and finds directions of high variation. The first two components provide a 2D projection for the scatter view. An item appearing near another item in the plot can be a useful visual clue, but dimensionality reduction discards information. Full-space neighbors need not remain neighbors on the screen.

The synthetic categories make the demonstration approachable. The figure does not demonstrate the quality of a real embedding model, and the app does not train a neural model using PCA.

## 9. Document processing and retrieval

### 9.1 Import and chunking

Text enters through a file importer or a paste form. The title identifies the document; chunk numbers locate passages within that document. The default chunk size is 180 words with 35 words of overlap. Chunk overlap helps retain information near boundaries. It can also produce similar adjacent hits, so users should review context rather than treating every hit as independent evidence.

PDF extraction is best effort. PDF is a layout format, and text reading order can be imperfect. Scans contain page images rather than selectable text and require OCR. Tables, equations and complex columns may need manual cleanup before retrieval is useful.

### 9.2 TF-IDF

Term frequency reflects how often a term occurs in a chunk. Inverse document frequency downweights terms that appear in many chunks. The resulting weighted vectors are compared to the query. The same corpus vocabulary and weighting convention must be used for both.

This method is understandable and does not need model downloads. It is sensitive to vocabulary: an answer described using different words may not rank highly. Language-specific tokenization, stemming and morphology are future extension points. Unicode text support alone should not be interpreted as tested multilingual retrieval quality.

### 9.3 Cited extracts

Offline output presents relevant passages with source labels. A citation identifies the supplied text, not independent verification that the source is correct. The user can compare the extract with the full document. When the data does not support a useful response, the application should not fabricate a fact.

### 9.4 Retrieval-augmented generation

The optional RAG workflow is: embed the question, retrieve document chunks, construct a prompt containing those chunks, ask the local generation model and display its response with retrieved evidence. The model is not being retrained or fine-tuned. This project retrieves context at request time.

Model-generated text requires review. Context limits may exclude relevant material, retrieval may choose the wrong passage, and the model may make claims not justified by the passages. Adding citations to an answer does not by itself prove that every statement is supported.

## 10. Installation and first-run procedure

### Windows

1. Install Python from python.org with Tcl/Tk support. Enable PATH integration when the installer offers it.
2. Extract the complete ZIP to a writable folder, such as a folder under Documents.
3. Double-click `START_WINDOWS.bat`, or open PowerShell in the extracted folder.
4. Run `py -3 app.py` or `python app.py`.
5. Use the offline demo before installing optional tools.

### macOS and Linux

Run `python3 --version`, then `python3 -m tkinter`. If the Tk test opens a small window, run `python3 app.py`. The launcher can also be run with `bash start.sh`. On Debian/Ubuntu systems missing Tk, the matching package is commonly `python3-tk`; other distributions use their own packaging.

A GUI session is required for Tkinter windows. On a headless server, run the application's API mode. The launch scripts are convenience wrappers around Python; they are not standalone executables that bundle an interpreter.

### Optional dependencies

Use `python -m pip install -r requirements-pdf.txt` for text PDF import. Use the same interpreter name that starts the app, such as `python3`. Ollama installation and model downloads are separate from pip. The presentation PDF is already included; `reportlab` is needed only for regenerating it.

## 11. Configure and run optional Ollama

Install Ollama from its official site. Download `nomic-embed-text` and `llama3.2:1b` using `ollama pull`, then use `ollama list` to confirm that they are present. Start `ollama serve` only if the service is not already running.

The default local base URL is `http://localhost:11434`. The API provides `/api/tags` to list models, `/api/embed` to generate vectors and `/api/generate` to create text. The application does not need a paid key for this local configuration. The model names in the application must exactly match installed model names.

The initial installation and model download need network access. Subsequent local inference can operate without a cloud service. Memory, storage and latency depend on the actual downloaded model and hardware; this report does not promise a universal RAM minimum or fixed response time. A CPU can make small-model inference possible, but a request may still take noticeable time.

## 12. Storage, backup and maintenance

SQLite stores documents and their chunks in a file on the local machine. Model metadata and embedding compatibility must be retained when vectors are persisted. Changing the embedding model requires compatible vectors rather than mixing old and new ones. The application invalidates incompatible cached embeddings and rebuilds them automatically on the next Ollama search.

The application is designed for a single user. Close it before manually copying its data directory for a backup, and keep exported files in a safe location. Local storage is not automatically encrypted. Deleting an item in the UI is an application operation, not a secure-erasure guarantee for copies, filesystem snapshots or previous backups.

The default data directory is `.Rax_ai` under the current user's home folder. It contains `knowledge.sqlite3` and `settings.json`. Override it with `python app.py --data-dir ./my-data`. Vector-lab changes are session-only. **Documents > Export backup** writes document JSON; **Import backup** merges by title/text identity, safely ignores supplied IDs and rebuilds chunks. Backups do not include cached embeddings or model weights. Imports allow files up to 10 MB and text up to 200,000 characters per document. JSON backups are limited to 50 MB, 1,000 documents and 10 million total text characters.

Keep imported documents reasonably sized for a desktop teaching tool. A production system would need more sophisticated memory management, index maintenance, document versioning, access control and backup/recovery processes.

### 12.1 Database schema

| Table | Main fields | Relationship / purpose |
|---|---|---|
| `documents` | `id`, `title`, `text`, `created_at`, `fingerprint` | One complete document; unique title/text fingerprint |
| `chunks` | `id`, `document_id`, `ordinal`, `text` | Many chunks per document; cascades when parent is deleted |
| `embeddings` | `chunk_id`, `model_key`, `text_hash`, `vector_json` | Model-specific cached vectors per chunk |

Database writes use transactions and foreign keys. A chunk deletion through removal of its parent document also removes its cached embeddings. Settings use a separate JSON file rather than mixing GUI preferences with document text.

### 12.2 Source module map

| File | Responsibility |
|---|---|
| `app.py` | Parse command-line flags; select GUI, API or self-check |
| `your_ai/gui.py` | Home, Documents, Ask documents, Vector lab and Settings pages |
| `your_ai/vectors.py` | Distance validation, brute force, KD-Tree, HNSW, PCA and demo benchmarks |
| `your_ai/knowledge.py` | Import, chunking, persistence, TF-IDF, source extraction and semantic retrieval |
| `your_ai/ollama_client.py` | Local-model discovery, embedding and generation requests |
| `your_ai/settings.py` | Default preferences and atomic settings storage |
| `your_ai/sample_data.py` | Built-in educational sample documents |
| `your_ai/api.py` | Authenticated, loopback-only JSON development API |
| `tests/` | Automated validation without a downloaded model |

### 12.3 Local API contract

Start with `python app.py --api --port 8765`. A random bearer token is printed on startup. An explicit token can be supplied with `--api-token`, but avoid storing sensitive tokens in shared scripts or shell history. Use `Authorization: Bearer <token>` for every `/api/` route. A health check at `/health` is available without that header. JSON request bodies use `Content-Type: application/json` and must be at most 1 MiB. Foreign browser origins are rejected.

| Method | Route | Input / purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/stats` | Knowledge and vector statistics |
| GET | `/api/documents` | List documents |
| POST | `/api/documents` | `title`, `text`; add a document |
| DELETE | `/api/documents/<id>` | Delete a document and its chunks |
| POST | `/api/search` | `query`, optional `k`, `mode=offline`; find passages |
| POST | `/api/ask` | `question`, optional `k`, `mode=offline`; return extracts |
| GET | `/api/vectors` | List session vectors |
| POST | `/api/vectors` | `label`, `category`, `vector`; add a session vector |
| DELETE | `/api/vectors/<id>` | Remove a session vector |
| POST | `/api/vector-search` | `query`, optional `k`, `metric`, `algorithm` |
| POST | `/api/benchmark` | `query`, optional `k`, `metric`, `repeats` |

Vector queries may be a supported demo label or a numeric array. Algorithms are `hnsw`, `kd_tree` and `brute_force`; metrics are `cosine`, `euclidean` and `manhattan`. API `k` ranges from 1 to 20; benchmark repeats from 1 to 25. The API document-answer endpoints currently support offline mode only. The GUI provides optional Ollama mode. A fresh API-only database is empty until you add documents; the desktop's first-run sample initialization is separate.

## 13. Benchmark methodology

The benchmark compares actual implementation work, not hard-coded times. It uses a fixed dataset, common query and metric, measures elapsed execution using a high-resolution timer and compares approximate neighbors with the exact reference. Results depend on dataset size, dimensionality, graph parameters, Python version, CPU load and warm-up state.

Recall@k is the fraction of exact top-k neighbors recovered by the tested algorithm, using the benchmark's tie convention. For example, recovering 4 of 5 exact neighbors gives recall@5 of 0.8. It does not measure how correct an answer is or how useful the documents are. Precision of natural-language answers needs a separate labelled evaluation set.

Report build time and query time separately. A fast query can require a costly index build. Small example data may be dominated by Python or GUI overhead. Do not infer million-document throughput, energy savings or organization-wide productivity gains from this demo.

## 14. Validation and reproducibility

The package contains automated tests that can run with `python -m unittest discover -s tests -v`. Algorithm tests should compare exact results across metrics and check dimensions, ties and invalid values. Document tests should cover persistence, chunk deletion, import behavior and lexical retrieval. API and local-model client tests should verify request contracts and error handling without requiring real model downloads.

The completed validation run passed **61 automated tests** on **Linux x86_64 with Python 3.12.14**:

| Area | Passing tests | Scope |
|---|---:|---|
| Vector engine | 10 | Exact metrics, KD-Tree comparison, real HNSW layers, recall, mutations and PCA |
| Local API | 9 | Actual HTTP requests, CRUD, authentication, host/origin checks and malformed input |
| Documents and Ollama client | 35 | Persistence, Unicode, backups, import, semantic caching, errors and mocked model requests |
| GUI controllers | 7 | Widget stand-ins with real SQLite and background-worker behavior |

`python app.py --self-check`, the shell launcher's `--self-check`, and `python examples/offline_demo.py` passed. All project Python files parsed using Python 3.10 syntax rules, which is not equivalent to running them under Python 3.10. Optional PDF import was exercised with pypdf 6.10.0 using the generated 24-page presentation; extracted text was searchable and returned matching passages for a Tkinter/SQLite query.

Windows and macOS launchers were inspected, but those operating systems were not available for execution. Real Ollama models were unavailable, so model APIs were tested using mocks; real inference speed and answer quality remain unverified. The delivered validation record is also summarized in the presentation. See `TESTING.md` for reproducible commands and the consolidated validation record. A real GUI launch could not be checked in the build environment because desktop-display socket creation was unavailable. A manual desktop checklist is supplied for validation on the user's computer. No screenshot in this package is presented as evidence of a verified desktop launch. Mock Ollama integration tests validate application behavior around a simulated service; they do not verify a real model's performance or answer quality.

For repeatable evaluation, retain the Python version, dataset, query set, random seed, metric, k value and graph parameters. Compare query medians or a repeated sample rather than interpreting one very small timing as definitive.

## 15. Known limitations and appropriate use

- Educational Python implementations are intended for small to moderate personal experiments, not production-scale indexing.
- Offline retrieval matches weighted terms; it has limited synonym and morphology awareness.
- Synthetic demo vectors illustrate mechanisms, not learned semantic understanding.
- PCA is a lossy visualization.
- HNSW can miss true nearest neighbors.
- PDF import needs a text layer and may lose reading order.
- Local model results depend on the model and can contain unsupported claims.
- The local API requires a bearer token, but the application has no multi-user accounts, role-based permissions or built-in encryption.
- A public deployment would require a separate security and operational design.
- Windows and macOS instructions are provided, but validation must be described according to the actual tested environment.

These boundaries make the project suitable for learning, a student demonstration, personal note exploration and controlled prototypes. They are not reasons to misrepresent it as a smaller version of an unbounded chat model.

## 16. Future scope and expected impact

The following are proposed extensions, not delivered features:

1. **Better retrieval quality:** hybrid lexical/dense ranking, reranking, query expansion and a labelled question/answer evaluation set.
2. **Richer documents:** OCR, DOCX support, section-aware chunking, page coordinates and clickable passage highlighting.
3. **Scale and performance:** optimized numeric arrays, production ANN backends, batch embeddings and incremental index updates.
4. **Responsible shared use:** authentication, encryption, access-aware retrieval, audit logs and managed backups.
5. **Distribution:** a signed standalone installer and platform-specific testing.
6. **Language support:** tokenizer and embedding evaluation for Marathi, Hindi and other selected languages.
7. **Usability:** keyboard accessibility, saved queries, clearer document version history and source comparison.

Potential impact includes easier access to personal notes, lower setup barriers for vector-search education and a transparent base for document assistants. Benefits should be measured through relevant outcomes: retrieval recall on a labelled corpus, task-completion time, source-checking accuracy and user feedback. No measured business impact is claimed in this release.

## 17. Five-minute presentation script

- **0:00-0:40:** Introduce the problem: finding useful information in notes and understanding vector search without complex setup.
- **0:40-1:30:** Search the synthetic vector demo, show distance choices and explain that full vectors determine neighbors.
- **1:30-2:10:** Compare algorithms and explain the recall/latency trade-off.
- **2:10-3:20:** Import a sample note, ask a related question and inspect the returned source extracts.
- **3:20-4:10:** Explain the optional Ollama branch, showing its configuration only if the models are installed.
- **4:10-5:00:** Explain persistence, testing, known limitations and one prioritized future extension.

## 18. Questions and answers for presentation

**Is this a newly trained AI model?** No. The project implements retrieval and connects to optional pretrained local models.

**Why Python?** It makes the algorithms and application flow easier to read and removes the need to compile C++ for this project.

**Why three algorithms?** They allow a meaningful comparison between exact scanning, exact spatial pruning and approximate graph search.

**Does it work without internet?** Core vector search and lexical document extraction work without network access once Python is installed. Ollama also needs initial installation and model downloads.

**Why can brute force sometimes win?** Tiny datasets have little search work, so tree or graph overhead may cost more than a direct scan.

**What is the difference between retrieval and generation?** Retrieval selects existing passages. Generation writes new text conditioned on a prompt and retrieved context.

**Do citations guarantee truth?** No. They help trace evidence, but the original source and the model's use of it still need review.

**Why SQLite?** It gives a simple durable local store without requiring users to install or manage a database server.

**What is the strongest next improvement?** Build a representative evaluation dataset, measure retrieval quality, then add improvements that show a measurable benefit.

## 19. Glossary

- **Embedding:** a vector produced from input by a model.
- **Vector index:** a structure used to search stored vectors.
- **Nearest neighbor:** an item with a small distance to a query under the selected metric.
- **Chunk:** a segment of a document used as a retrieval unit.
- **TF-IDF:** term frequency multiplied by inverse document frequency.
- **PCA:** a projection that captures selected directions of variation.
- **HNSW:** Hierarchical Navigable Small World graph.
- **RAG:** retrieval-augmented generation.
- **Recall@k:** overlap with reference top-k neighbors, divided by the reference count.
- **Loopback:** a network address referring to the same computer, such as 127.0.0.1.

## 20. References and attribution

See `REFERENCES.md` for primary technology documentation and research references. The supplied original archive is the project concept source. Implementation-specific claims should be checked against the delivered source and test evidence, rather than inferred from external documentation.


## Appendix: manual desktop verification checklist

Run these checks in a normal desktop session after extracting the ZIP. This is a checklist for the user, not a claim that live-window testing occurred in the build environment.

1. Run `python -m tkinter`; verify that its small test window opens, then close it.
2. Run `python app.py`; verify that Home opens and the five sidebar pages are visible.
3. Open Documents and confirm sample notes are present on first launch.
4. Add `examples/my_first_document.txt` through Import file, then select it and read the preview.
5. Ask a related question in Offline mode; inspect the answer and each source row.
6. Use Find passages and Export answer; open the exported text file.
7. Export a backup, import it again and verify duplicates are not multiplied.
8. In Vector lab, search `binary tree`, compare all algorithms and change the distance metric.
9. Add and delete a demo vector using Manage vectors; remember these edits are session-only.
10. Close and reopen the application; verify document persistence.
11. If using Ollama, install both models, check the connection in Settings and ask in Ollama mode. Confirm a real answer and inspect source evidence.
12. If PDF import is needed, install the optional dependency and import a PDF with selectable text. Confirm readable content and useful retrieval.

If a step fails, run the application from a terminal, note the error and consult README troubleshooting. No administrator privileges are required to run the project from a writable user folder.
