#!/usr/bin/env python3
"""Launch the desktop app, optional localhost API, or diagnostic checks."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rax Ai - Python desktop vector search and document assistant")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".Rax_ai", help="Folder for saved documents and settings")
    parser.add_argument("--api", action="store_true", help="Run the optional local REST API instead of the desktop UI")
    parser.add_argument("--port", type=int, default=8765, help="Local API port (default: 8765)")
    parser.add_argument("--api-token", help="Optional API bearer token; otherwise a fresh token is generated")
    parser.add_argument("--self-check", action="store_true", help="Check offline functionality without opening a window")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required. Python 3.12 is recommended.")
    from your_ai.knowledge import KnowledgeStore
    from your_ai.vectors import VectorLab
    if args.self_check:
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            store = KnowledgeStore(Path(folder) / "check.sqlite3")
            store.load_samples()
            assert store.stats()["documents"] > 0
            store.add_document("Self-check", "Python stores structured records in an SQLite database.")
            assert store.search("SQLite database", k=3)
            assert store.ask("SQLite database")["sources"]
            lab = VectorLab()
            assert len(lab.search(lab.items[0]["vector"], k=3)["results"]) == 3
            for row in lab.benchmark(lab.items[0]["vector"], k=3):
                assert 0 <= row["recall_at_k"] <= 1
        print("PASS: database, documents, offline search/answers, three vector indexes and benchmark.")
        try:
            import tkinter
            print("Tkinter imports successfully. A graphical desktop is required to open the UI.")
        except ImportError:
            print("Tkinter is missing. See README: install Python with Tcl/Tk, or python3-tk on Linux.")
        print("Optional Ollama models are not checked by --self-check. Use Settings > Check connection.")
        return 0
    args.data_dir = args.data_dir.expanduser().resolve()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(args.data_dir / "knowledge.sqlite3")
    lab = VectorLab()
    if args.api:
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        from your_ai.api import serve
        serve(store, lab, port=args.port, token=args.api_token)
        return 0
    try:
        import tkinter as tk
        from your_ai.gui import Application
        root = tk.Tk()
    except (ImportError, RuntimeError) as exc:
        print(f"Cannot open the desktop interface: {exc}\nSee README for Tkinter installation.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Cannot open the desktop interface: {exc}\nRun on a computer with a graphical desktop. For headless checks, use python app.py --self-check.", file=sys.stderr)
        return 1
    Application(root, store, lab, args.data_dir)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        raise SystemExit(1)
