# Validation record

The release was checked with Python 3.12.14 on Linux x86_64.
All 61 automated tests passed. Tests use temporary data and do not modify the
user's saved collection.

| Area | Checks | Result |
|---|---:|---|
| Vector algorithms and PCA | 10 | Passed; KD-tree checked against exact brute force, HNSW recall measured on fixed test data |
| Document storage and Ollama client | 35 | Passed; transactions, Unicode, backups, source extracts, invalid inputs and mocked model responses |
| Local REST API | 9 | Passed using actual localhost HTTP requests |
| Desktop controller integration | 7 | Passed with widget stand-ins and real SQLite, vector indexes and worker threads |

Additional checks passed: command-line self-check, Unix shell launcher with
`--self-check`, syntax compilation, an offline demonstration, and actual
text extraction/search of the 24-page presentation PDF using pypdf 6.10.0.
Python 3.10 syntax was checked; the full runtime suite was run on Python 3.12.

## Validation limits

- A real desktop window could not be rendered in the build environment because
  a virtual display could not create its required socket. Controller tests are
  not a substitute for visual testing of Tk layout, fonts and dialogs.
- The Windows launcher was reviewed but not executed on Windows. macOS was
  not available for platform testing.
- No Ollama model was installed in the build environment. Model API behavior
  is tested with controlled responses; real model inference, answer quality,
  hardware performance and model installation remain machine-dependent.
- Tiny example benchmarks do not establish performance at production scale.

## Reproduce the checks

From the extracted project folder:

```bash
python app.py --self-check
python -m unittest discover -s tests -v
python examples/offline_demo.py
```

Use `py -3` instead of `python` on Windows if needed, or `python3` on macOS/Linux.
Tkinter controller checks are skipped if the Tkinter module is not installed.
The algorithm, document and API tests do not require a display or downloaded models.

## Desktop demonstration checklist

1. Run `python app.py`. Verify that all five navigation pages are readable and
   the window can be resized on the presentation computer.
2. On Home, click **Ask a sample question**. Confirm that the offline answer
   explains RAG and that its numbered source passage opens.
3. In Documents, import `examples/my_first_document.txt`. Select it and confirm
   the full text appears. Ask: `When is the Python project presentation?`
4. Close and reopen the app. Confirm the imported document remains saved.
5. Export a JSON backup. Delete that imported document, then import the backup
   and verify the document returns. Existing documents should remain intact.
6. In Vector lab, search `binary tree` using each distance and algorithm. Click
   **Compare algorithms** and inspect measured time and Recall@K.
7. Add a custom 16-number vector with **Manage vectors**; search it and delete
   it. These lab edits intentionally reset on restart.
8. Optional: install Ollama and the two specified models, check the connection,
   and repeat a document question in **Ollama (local AI)** mode. Review citations.

For the optional API, run `python app.py --api`, keep its printed token private,
and call `python examples/api_client.py --token YOUR_PRINTED_TOKEN` in a second
terminal. Stop the server with Ctrl+C.
