"""A reproducible demo that also works without a desktop display.

From the project folder: python examples/offline_demo.py
This creates a temporary database and does not modify your saved collection.
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from your_ai.knowledge import KnowledgeStore
from your_ai.vectors import VectorLab


def main():
    with tempfile.TemporaryDirectory() as folder:
        store = KnowledgeStore(Path(folder) / "demo.sqlite3")
        store.load_samples()
        question = "What is RAG?"
        response = store.ask(question)
        print("Rax Ai - OFFLINE DEMONSTRATION\n")
        print("Question:", question)
        print("\n" + response["answer"])
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f'  [{i}] {source["title"]}')
        print("\nVECTOR LAB - binary tree, cosine distance, top 5")
        lab = VectorLab()
        for item in lab.search("binary tree")["results"]:
            print(f'  {item["label"]:25s} distance={item["distance"]:.5f}')
        print("\nMeasured benchmark (build time excluded; this tiny dataset is educational):")
        for row in lab.benchmark("binary tree"):
            print(f'  {row["algorithm"]:12s} {row["mean_ms"]:.4f} ms   recall={row["recall_at_k"]:.0%}')
        store.close()


if __name__ == "__main__":
    main()
