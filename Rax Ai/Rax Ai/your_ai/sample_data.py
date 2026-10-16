"""Original small documents included for a repeatable offline demonstration."""

SAMPLE_DOCUMENTS = [
    {
        "title": "Welcome to Rax Ai",
        "text": """Rax Ai is a private desktop document assistant written in Python. Its simple interface lets you add documents, search passages, ask questions, and manage a local knowledge library.

Offline mode works immediately without model downloads or internet access. It uses TF-IDF keyword retrieval and returns exact source excerpts with numbered citations. Offline mode does not generate new text or understand synonyms like a language model. If your question shares no useful words with your documents, it reports that no matching evidence was found.

Local AI mode connects only to Ollama running on your own computer. The embedding model converts document chunks and a question into vectors. The app retrieves similar passages, then a separate language model writes an answer based on those passages. The model may still make mistakes, so always check its cited source text.

Your documents are stored in a local SQLite database. Backups are readable JSON files containing document titles and text. Importing a backup merges documents and skips identical duplicates. Derived chunks and embeddings are rebuilt instead of trusting backup index data.

To try the app, load the sample documents and ask: How does offline mode work? You can also search for SQLite, vector database, or solar energy.""",
    },
    {
        "title": "Vector Search and RAG Explained",
        "text": """A vector database stores numerical vectors and retrieves nearby vectors using a chosen distance or similarity measure. A vector is an ordered list of numbers. Text embedding models convert words, sentences, and documents into numerical vectors. Texts with related meanings often have nearby vectors, although quality depends on the model and language.

Cosine similarity compares the directions of two vectors. For normalized vectors, their dot product is their cosine similarity. Values closer to one indicate more similar directions. Euclidean distance measures the straight-line distance between vectors.

HNSW means Hierarchical Navigable Small World. It builds a graph in multiple layers so a search can navigate toward nearby vectors. HNSW is an approximate nearest-neighbor method: it can be faster on large collections, but a query is not guaranteed to find every exact nearest neighbor. Parameters such as M and ef_search trade memory and speed against recall.

RAG means Retrieval-Augmented Generation. First, a retriever finds relevant source passages. Next, a language model receives the question and those passages and produces a grounded answer. Citations make it easier to inspect evidence. RAG does not automatically guarantee accuracy or eliminate hallucinations.

Chunking divides a long document into smaller passages. A modest overlap retains words near boundaries. Smaller passages improve focused retrieval, while larger passages preserve more context. The current application uses windows of up to 180 whitespace-separated words with a 35-word overlap.""",
    },
    {
        "title": "Solar Energy Study Notes",
        "text": """Solar panels convert sunlight into electricity using photovoltaic cells. An inverter converts the direct current produced by panels into alternating current for many household appliances.

Solar output depends on sunlight, panel orientation, shading, temperature, and maintenance. Trees and nearby buildings can shade a panel and reduce its electricity production. A battery stores some energy for later use, but it adds cost and does not provide unlimited backup.

A grid-connected system can use the electricity grid when solar production is insufficient. A standalone system must be designed around local sunlight, daily energy needs, and storage capacity. System costs and local electricity rules vary, so a qualified installer should review a real installation.

Renewable energy education helps students understand power generation, efficiency, and environmental trade-offs. Solar panels produce electricity without burning fuel during operation. Manufacturing, transport, maintenance, and end-of-life handling still have environmental impacts.""",
    },
    {
        "title": "Study Notes: Python and Local Data",
        "text": """Python is a general-purpose programming language used for automation, desktop applications, data analysis, and artificial intelligence. Readable syntax and a large ecosystem make it suitable for learning as well as practical tools.

Tkinter provides a Python interface to the Tk graphical toolkit. It can create windows, buttons, tabs, tables, text fields, and file-selection dialogs. Long-running work should run outside the UI event loop so the application remains responsive.

SQLite is an embedded relational database. It stores data in a local file and supports SQL queries and transactions. A transaction makes a group of changes succeed together or roll back together when an error occurs.

The Python standard library includes sqlite3 for database access, json for portable backups, urllib for HTTP requests, unittest for automated testing, and threading for background tasks. PDF text extraction is optional and uses the additional pypdf package. Scanned images need OCR, which this application does not provide.

Unicode supports text from multiple writing systems. The application stores and searches Unicode text, including English, Hindi, and Marathi. Example: ज्ञान हेच खरे धन आहे. Offline keyword search depends on matching words and does not translate between languages.""",
    },
]
