"""Correctness tests for exact indexes, measured ANN recall, and playground CRUD."""

import math
import random
import unittest

from your_ai.vectors import BruteForce, HNSW, KDTree, METRICS, VectorLab, distance, pca_project


class ExactSearchTests(unittest.TestCase):
    def test_kd_matches_brute_force_random_vectors_all_metrics(self):
        rng = random.Random(8731)
        for dimension in (1, 3, 16):
            items = {i: [rng.uniform(-2, 2) for _ in range(dimension)] for i in range(83)}
            items[90] = [0.0] * dimension
            brute, kd = BruteForce(items), KDTree(items)
            for _ in range(8):
                query = [rng.uniform(-2, 2) for _ in range(dimension)]
                for metric in METRICS:
                    for k in (1, 5, 100):
                        with self.subTest(d=dimension, metric=metric, k=k):
                            self.assertEqual(brute.search(query, k, metric), kd.search(query, k, metric))

    def test_cosine_zero_vectors_and_ties(self):
        items = {7: [0, 0], 2: [1, 0], 9: [2, 0], 3: [-1, 0], 5: [0, 1], 1: [0, 0]}
        brute, kd = BruteForce(items), KDTree(items)
        self.assertEqual(brute.search([0, 0], 3), [(1, 0), (7, 0), (2, 1)])
        for q in ([0, 0], [1, 0], [-1, 0], [1, 1]):
            for metric in METRICS:
                self.assertEqual(brute.search(q, 10, metric), kd.search(q, 10, metric))
        self.assertEqual(distance([0, 0], [0, 0]), 0)
        self.assertEqual(distance([0, 0], [1, 0]), 1)
        self.assertEqual(distance([1, 0], [-1, 0]), 2)

    def test_cosine_pruning_is_normalized(self):
        # Raw-coordinate pruning would drop genuinely closer points here.
        items = {0: [1, 5], 1: [1000, 1], 2: [100, 0], 3: [-2, 1], 4: [0, 0], 5: [2, 1]}
        self.assertEqual(KDTree(items).search([1, 0], 3), BruteForce(items).search([1, 0], 3))

    def test_validation_and_empty_index(self):
        for cls in (BruteForce, KDTree, HNSW):
            for invalid in ({1: [1, 2], 2: [1]}, {1: [float("nan")]}, {1: [float("inf")]},
                            {1: []}, {1: [True]}, {1: ["1"]}, {"bad": [1]}, {1: [1e308]}):
                with self.subTest(index=cls.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        cls(invalid)
            index = cls({1: [1, 2]})
            for q in ([1], [], [math.nan, 0], "1,2"):
                with self.assertRaises(ValueError):
                    index.search(q)
            for k in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    index.search([1, 2], k)
            with self.assertRaises(ValueError):
                index.search([1, 2], metric="unsupported")
            self.assertEqual(cls({}).search([1, 2]), [])


class ApproximateSearchTests(unittest.TestCase):
    def test_hnsw_real_layers_and_empirical_recall(self):
        rng = random.Random(58)
        items = {i: [rng.uniform(-1, 1) for _ in range(8)] for i in range(140)}
        hnsw, brute = HNSW(items, ef_search=48), BruteForce(items)
        queries = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(10)]
        for metric in METRICS:
            recalls = []
            for q in queries:
                exact = {i for i, _ in brute.search(q, 7, metric)}
                found = hnsw.search(q, 7, metric)
                self.assertEqual(found, sorted(found, key=lambda pair: (pair[1], pair[0])))
                self.assertEqual(len({i for i, _ in found}), len(found))
                self.assertTrue(all(i in items and math.isfinite(d) for i, d in found))
                recalls.append(len(exact & {i for i, _ in found}) / 7)
            # A regression gate on this fixed test dataset, not a universal guarantee.
            self.assertGreaterEqual(sum(recalls) / len(recalls), 0.85)
            info = hnsw.info(metric)
            self.assertGreater(info["layer_count"], 1)
            self.assertEqual(info["layers"][0]["nodes"], len(items))
            self.assertGreater(info["layers"][0]["directed_edges"], 0)
            for layer in info["layers"]:
                limit = 16 if layer["level"] == 0 else 8
                self.assertLessEqual(layer["directed_edges"], layer["nodes"] * limit)

    def test_fixed_seed_determinism(self):
        items = {i: [math.sin(i), math.cos(i)] for i in range(35)}
        one, two = HNSW(items), HNSW(dict(reversed(list(items.items()))))
        self.assertEqual(one.search([0.4, 0.7]), two.search([0.4, 0.7]))
        self.assertEqual(one.info(), two.info())


class VectorLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = VectorLab()

    def test_demo_query_and_projection(self):
        self.assertEqual(len(self.lab.items), 40)
        self.assertEqual(self.lab.info()["dimensions"], 16)
        for label in ("binary tree", "calculus", "pizza", "cricket"):
            result = self.lab.search(label, algorithm="brute_force")
            self.assertEqual(result["results"][0]["label"], label)
            self.assertIn("synthetic", result["query_mode"])
            self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertEqual(self.lab.search("find sushi", algorithm="brute_force")["results"][0]["label"], "sushi")
        self.assertEqual(self.lab.search("Food", algorithm="brute_force")["results"][0]["category"], "Food")
        with self.assertRaisesRegex(ValueError, "No demo label matched"):
            self.lab.search("arbitrary unseen sentence")
        projection = self.lab.project()
        self.assertEqual(projection, self.lab.project())
        self.assertEqual(len(projection), 40)
        self.assertTrue(all(math.isfinite(p[axis]) for p in projection for axis in ("x", "y")))

    def test_benchmark_recall_is_measured(self):
        rows = self.lab.benchmark("pizza", repeats=2)
        baseline = {result["id"] for result in next(r for r in rows if r["algorithm"] == "brute_force")["results"]}
        for row in rows:
            found = {item["id"] for item in row["results"]}
            self.assertEqual(row["recall_at_k"], len(found & baseline) / len(baseline))
            self.assertGreaterEqual(row["mean_ms"], 0)
            if row["algorithm"] != "hnsw":
                self.assertEqual(row["recall_at_k"], 1)

    def test_mutations_rebuild_all_indexes(self):
        lab = VectorLab([{ "id": 1, "label": "one", "category": "test", "vector": [1, 0]}])
        new_id = lab.add("two", "test", [0, 1])
        for algorithm in ("hnsw", "kd_tree", "brute_force"):
            self.assertEqual(lab.search("two", algorithm=algorithm)["results"][0]["id"], new_id)
        with self.assertRaises(ValueError):
            lab.add("TWO", "test", [1, 1])
        with self.assertRaises(ValueError):
            lab.add("bad", "test", [1])
        with self.assertRaisesRegex(ValueError, "rescale"):
            lab.add("overflow", "test", [1e308, -1e308])
        self.assertEqual(len(lab.items), 2)
        lab.delete(new_id)
        self.assertEqual(len(lab.project()), 1)
        with self.assertRaises(ValueError):
            lab.delete(new_id)
        lab.delete(1)
        self.assertEqual(lab.search([0, 1])["results"], [])
        self.assertEqual(lab.project(), [])
        self.assertEqual(lab.info()["dimensions"], 2)
        self.assertGreater(lab.add("new", "test", [1, 1]), new_id)

    def test_pca_degenerate_and_known_axis(self):
        self.assertEqual(pca_project([]), [])
        self.assertEqual(pca_project([[2, 2], [2, 2]]), [(0, 0), (0, 0)])
        points = pca_project([[-2, 0], [0, 0], [2, 0]])
        self.assertAlmostEqual(points[0][0], -2)
        self.assertAlmostEqual(points[2][0], 2)
        self.assertTrue(all(abs(y) < 1e-10 for _, y in points))
        self.assertEqual(pca_project([[0], [2]]), [(-1, 0), (1, 0)])


if __name__ == "__main__":
    unittest.main()
