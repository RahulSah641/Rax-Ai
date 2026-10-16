# References and technical grounding

These sources explain external technologies or algorithm concepts. The Python project implementation and its validation records are the source for claims about what this delivered application does. External documentation does not certify this application. Sources checked on 9 September 2026.

1. **Python Software Foundation - Tkinter.** Desktop toolkit, event loop and installation check. https://docs.python.org/3/library/tkinter.html
2. **Python Software Foundation - sqlite3.** Python interface to SQLite. https://docs.python.org/3/library/sqlite3.html
3. **Python Software Foundation - unittest.** Automated test framework. https://docs.python.org/3/library/unittest.html
4. **Ollama - API introduction.** Local base URL and API overview. https://docs.ollama.com/api/introduction
5. **Ollama - Generate embeddings.** Current `/api/embed` request and response. https://docs.ollama.com/api/embed
6. **Ollama - Generate a response.** `/api/generate`, prompts and generation options. https://docs.ollama.com/api/generate
7. **Ollama - List models.** `/api/tags` and available model metadata. https://docs.ollama.com/api/tags
8. **Ollama - Quickstart.** Local installation and command-line usage. https://docs.ollama.com/quickstart
9. **Ollama model catalog - nomic-embed-text.** Optional embedding model. https://ollama.com/library/nomic-embed-text
10. **Ollama model catalog - llama3.2:1b.** Optional generation model. https://ollama.com/library/llama3.2:1b
11. **Malkov, Y. A. and Yashunin, D. A.** Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. https://arxiv.org/abs/1603.09320
12. **SciPy - KDTree documentation.** Tree search and high-dimensional limitations. SciPy is a reference, not a runtime dependency of this project. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
13. **Manning, C. D., Raghavan, P. and Schutze, H. - Introduction to Information Retrieval.** TF-IDF weighting. https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html
14. **Manning, C. D., Raghavan, P. and Schutze, H. - Introduction to Information Retrieval.** Queries as vectors and cosine scoring. https://nlp.stanford.edu/IR-book/html/htmledition/queries-as-vectors-1.html
15. **pypdf - Extract Text from a PDF.** Text extraction limits; scanned images need OCR. https://pypdf.readthedocs.io/en/stable/user/extract-text.html

## Project provenance

The supplied archive `RAX-AI-main.zip` contains a C++ vector database educational project with a browser interface and local Ollama integration. This deliverable is a Python reimplementation and extension of that concept. It is not a binary conversion of the uploaded program. The algorithm implementations are readable educational Python, not bindings to a production vector database. No model weights are included. See the project notices for attribution and redistribution details.
