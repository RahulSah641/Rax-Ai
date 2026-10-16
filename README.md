# Rax Ai - Python Desktop Edition

A local-first vector search and document assistant built in Python with a simple desktop interface. Learn how vector search works, compare three real algorithms, save documents, and find useful passages without creating a cloud account.

**Start here:** extract this entire folder, install Python 3.10 or newer (Python 3.12 is a recommended baseline), then run:

```bash
python app.py
```

On Windows, you can also double-click `START_WINDOWS.bat`. On macOS/Linux, run `bash start.sh`. A desktop display and a Python installation with Tkinter are needed for the graphical application.

## What works without extra packages

- A simple Tkinter desktop interface with sidebar navigation.
- Synthetic 16-dimensional demo vectors with categories and a 2D PCA view.
- Genuine Python HNSW, KD-Tree and brute-force nearest-neighbor search.
- Cosine, Euclidean and Manhattan distance comparisons.
- Measured search benchmarks and recall against an exact baseline.
- Import plain text or Markdown, or paste text directly.
- SQLite persistence for documents and chunks.
- Offline TF-IDF retrieval with source-labelled extracts.
- A local REST API for experimenting from scripts.

**Offline mode returns relevant text extracts. It does not generate new AI prose.** Neural embeddings and generated answers are an optional second mode through Ollama, running on your own computer. No LLM or model weights are bundled with this ZIP.

## Windows: first run

1. Download Python from https://www.python.org/downloads/. Include Tcl/Tk support and enable the installer option to add Python to PATH when offered.
2. Right-click the ZIP, choose **Extract All**, and open the extracted `Rax_ai_Python` folder. Do not run files directly inside the ZIP.
3. Double-click `START_WINDOWS.bat`, or open a terminal in the folder and run `py -3 app.py`.
4. Start with the Vector lab page. Documents and offline answers work immediately.

To check Python and Tkinter:

```powershell
py -3 --version
py -3 -m tkinter
py -3 app.py
```

If `python` works but `py` does not, replace `py -3` with `python`.

## macOS / Linux: first run

```bash
python3 --version
python3 -m tkinter
python3 app.py
```

You can also use `bash start.sh`; to run it directly, use `chmod +x start.sh` once, followed by `./start.sh`.

On macOS, the Python installer from python.org is a straightforward option. On Debian/Ubuntu, if the Tkinter check fails, install your distribution's matching Tk package, commonly:

```bash
sudo apt install python3-tk
```

For other distributions or package managers, use the Tk package that matches your Python interpreter. Installing Tkinter with `pip install tkinter` is not the fix. Headless machines should use the API rather than the desktop UI.

## Optional PDF import

Only text-based PDF files are supported. Install the extra dependency into the same Python environment that runs the app:

```bash
python -m pip install -r requirements-pdf.txt
```

For `python3` users, replace `python` with `python3`. Scanned pages require OCR outside this application. Password-protected, damaged or unusually laid-out PDFs may not import cleanly.

## A five-minute guided demo

1. Launch `python app.py` and open **Vector lab**.
2. Search a provided concept such as `binary tree`, `sushi` or `basketball`.
3. Select HNSW, KD-Tree and brute force in turn. Compare their neighbors and measured times.
4. Run the benchmark. Recall describes agreement with exact nearest neighbors, not answer correctness.
5. Open **Documents**. The first launch includes sample notes; use **Add document** to paste your own note or **Import file** for a file.
6. Open **Ask documents**, choose **Offline (exact extracts)** and click **Ask question**. Try `How does offline mode work?` with the sample notes.
7. Read the returned extracts and their source labels. Ask an unrelated question to see how little or no supporting evidence is handled.
8. Close and reopen the app to confirm saved documents remain available.

The bundled vectors are synthetic teaching data. Searching their labels is a demonstration of the vector algorithms, not a general natural-language search model.

## Optional local AI with Ollama

Ollama is a separate application. The default embedding and generation models are not included in this ZIP. Installation and initial model downloads require an internet connection and disk space. Processing can run locally after the models have been downloaded. Model speed and memory usage depend on hardware, model version and context length.

1. Install Ollama: https://ollama.com/download.
2. In a terminal, download the two models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama list
```

3. Ensure the Ollama service is running. If it is not already running, start `ollama serve` in a separate terminal.
4. Open **Settings**, use **Check connection** and then **Save settings** for the local service at `http://localhost:11434` and confirm both model names.
5. In **Ask documents**, choose **Ollama (local AI)** and ask a question. Needed embeddings are built and cached when this mode is used.
6. Always review returned sources. A language model can still produce an unsupported statement despite receiving retrieved context.

`nomic-embed-text` produces embeddings; it is not a chat model. `llama3.2:1b` generates the answer. A new embedding model requires compatible embeddings for documents and queries. The app invalidates incompatible cached embeddings and rebuilds them automatically on the next Ollama search.

The application calls Ollama using Python's standard HTTP tools. The `ollama` Python package is not required. The relevant local endpoints are `/api/tags`, `/api/embed` and `/api/generate`.

## Runnable examples

```bash
python examples/offline_demo.py
```

This uses a temporary database, asks `What is RAG?`, prints source extracts, searches the demo vectors and runs the benchmark. It works without a graphical display and leaves your saved documents unchanged.

- `examples/my_first_document.txt` is a ready-to-import text sample.
- `examples/api_client.py` shows authenticated requests to the optional local server; run it with `--help` for its arguments.

## Algorithms and honest expectations

| Algorithm | Role | Important limitation |
|---|---|---|
| Brute force | Exact reference; visits every vector | Distance work grows with item count and dimension |
| KD-Tree | Exact search with spatial pruning | Pruning can become ineffective at high dimensions |
| HNSW | Approximate search using a layered neighbor graph | Recall depends on data and graph/search settings |

There is no guaranteed logarithmic runtime, no promise HNSW will win on small datasets, and no claim of production-scale readiness. Python overhead can dominate tiny benchmarks. The 2D PCA plot is only a projection and cannot preserve all distances in the original vector space.

## Data and privacy

Documents and chunks are saved in `knowledge.sqlite3` in your home folder's `.Rax_ai` directory. Preferences are in `settings.json` alongside it. Choose a different location with `python app.py --data-dir ./my-data`. Use **Documents > Export backup** and **Import backup** for a document JSON backup/merge. Backups contain documents, not model files or cached embeddings. Close the app before copying the whole data directory manually. Vector lab additions/deletions are session-only and reset when you restart.

Offline retrieval does not need an external AI service. In Ollama mode, relevant input and context are sent to the configured local service. Use a trusted local installation. The database and exports are not encrypted by this application. It is intended for one person on a trusted computer, not as a public multi-user server.

## Optional local REST API

The GUI runs directly and does not require this server. To use the API:

```bash
python app.py --api --port 8765
```

The server prints a new bearer token. Keep it private and use it in the `Authorization` header for every `/api/` request. `/health` is an unauthenticated health check. The API is loopback-only, and document search/answers through the API use **offline mode only**; use the GUI for Ollama.

This Python example asks the running API for a sourced extract:

```python
import json
from urllib.request import Request, urlopen

# Replace with the token printed by the API at startup.
token = "PASTE_YOUR_TOKEN_HERE"
body = json.dumps({"question": "What is RAG?", "k": 3}).encode("utf-8")
request = Request(
    "http://127.0.0.1:8765/api/ask", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + token}, method="POST")
with urlopen(request, timeout=30) as response:
    print(json.load(response))
```

The desktop seeds sample notes on first launch. For a fresh API-only data directory, add documents with `POST /api/documents` before asking questions. See `docs/PROJECT_REPORT.md` for all routes. Stop the server with Ctrl+C.

## Run automated tests

From the extracted project folder:

```bash
python app.py --self-check
python -m unittest discover -s tests -v
```

The delivered build passed **61 tests on Linux with Python 3.12.14**, including actual HTTP API requests, document persistence, vector algorithms, mocked Ollama requests and GUI-controller tests with widget stand-ins. Both self-check launch paths and the offline example passed. Optional text-PDF import was also exercised.

**A live desktop window and real Ollama inference could not be verified in the build environment.** Windows/macOS launchers were inspected but not executed on those operating systems. Python 3.10 syntax was checked, not a Python 3.10 runtime. See the presentation/report and manual desktop checklist for the precise validation scope.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Python command not found | Install Python, reopen the terminal; try `py -3` on Windows or `python3` on macOS/Linux |
| No module named tkinter | Install Tcl/Tk support for the same Python interpreter; run `python -m tkinter` to check |
| Window closes immediately | Run `python app.py` in a terminal and read the error; make sure the full ZIP was extracted |
| No display / DISPLAY error | Use a desktop session for the GUI, or run API mode on the headless machine |
| PDF import unavailable | Install `requirements-pdf.txt` using the same interpreter as the app |
| PDF imports no text | The file may be scanned; use OCR first and import the resulting text |
| Offline search finds nothing | Import documents first; try words present in the notes and a more specific question |
| Ollama offline | Start Ollama, check its base URL and run `ollama list` |
| Model missing | Pull the exact model name shown in settings, then retry |
| Generated answers are slow | A first request may load the model; try the smaller default model or offline extracts |
| SQLite permission error | Extract the project to a user-writable folder; avoid protected system folders |
| API port already used | Choose a different port with `--port` |

## Documentation

- `docs/Project_Presentation.pdf` - professional presentation covering purpose, language, libraries, architecture, setup, demo, testing and future scope.
- `docs/PROJECT_REPORT.md` - detailed technical and user documentation.
- `docs/REFERENCES.md` - official technology and research references.
- `docs/TESTING.md` - test evidence, limitations and a manual desktop checklist.

The application source is Python. Standard components such as Tk and SQLite may use native internals, but this project requires no C++ compilation. Optional models are separate downloads and have their own licenses and terms.
