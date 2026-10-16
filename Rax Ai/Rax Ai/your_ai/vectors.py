"""Educational, dependency-free vector indexes and a synthetic vector playground.

All rankings use ascending (distance, integer ID) for deterministic ties. Cosine
distance is 0 for two zero vectors and 1 when exactly one vector is zero.
HNSW is approximate: its recall and speed depend on data and search settings.
The tiny hand-designed demo vectors are NOT learned language embeddings.
"""

from __future__ import annotations

import heapq
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

METRICS = ("cosine", "euclidean", "manhattan")
ALGORITHMS = ("hnsw", "kd_tree", "brute_force")
MAX_COMPONENT = 1e100  # Keep distances and PCA covariance inside float64 range.


def validate_vector(vector: Sequence[float], dimension: int | None = None) -> list[float]:
    """Copy a nonempty, finite numeric vector and optionally check its dimension."""
    if isinstance(vector, (str, bytes)) or not isinstance(vector, (list, tuple)):
        raise ValueError("A vector must be a list of numbers.")
    if not vector or (dimension is not None and len(vector) != dimension):
        expected = f" exactly {dimension}" if dimension is not None else " at least one"
        raise ValueError(f"A vector must contain{expected} numeric values.")
    result = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Vector values must be numbers, not text or booleans.")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise ValueError("Vector values must be finite numbers.") from None
        if not math.isfinite(number):
            raise ValueError("Vector values must be finite numbers (no NaN or infinity).")
        if abs(number) > MAX_COMPONENT:
            raise ValueError("Vector magnitudes must be at most 1e100; rescale very large coordinates.")
        result.append(number)
    return result


def _metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError("Metric must be cosine, euclidean, or manhattan.")
    return metric


def _k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    return k


def _normalize(vector: Sequence[float]) -> list[float]:
    # Scale first: this avoids overflowing a sum of squared finite coordinates.
    scale = max((abs(x) for x in vector), default=0.0)
    if not scale:
        return [0.0] * len(vector)
    scaled = [x / scale for x in vector]
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    return [x / norm for x in scaled]


def distance(a: Sequence[float], b: Sequence[float], metric: str = "cosine") -> float:
    """Distance between validated vectors of the same dimension."""
    _metric(metric)
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension.")
    if metric == "euclidean":
        return math.dist(a, b)
    if metric == "manhattan":
        return math.fsum(abs(x - y) for x, y in zip(a, b))
    a_zero, b_zero = not any(a), not any(b)
    if a_zero or b_zero:
        return 0.0 if a_zero and b_zero else 1.0
    an, bn = _normalize(a), _normalize(b)
    return max(0.0, min(2.0, 1.0 - math.fsum(x * y for x, y in zip(an, bn))))


def _vectors(items: Mapping[int, Sequence[float]]) -> dict[int, list[float]]:
    if not isinstance(items, Mapping):
        raise ValueError("Index items must map integer IDs to vectors.")
    result: dict[int, list[float]] = {}
    dimension = None
    for item_id, vector in items.items():
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            raise ValueError("Vector IDs must be integers.")
        result[item_id] = validate_vector(vector, dimension)
        dimension = len(result[item_id])
    return result


class BruteForce:
    """Exact linear scan, used as the reference implementation."""

    def __init__(self, items: Mapping[int, Sequence[float]]):
        self.vectors = _vectors(items)
        self.dimension = len(next(iter(self.vectors.values()), []))

    def _query(self, query: Sequence[float], k: int, metric: str) -> list[float]:
        _k(k)
        _metric(metric)
        return validate_vector(query, self.dimension or None)

    def search(self, query: Sequence[float], k: int = 5, metric: str = "cosine") -> list[tuple[int, float]]:
        q = self._query(query, k, metric)
        ranked = sorted((distance(q, vector, metric), i) for i, vector in self.vectors.items())
        return [(i, d) for d, i in ranked[:k]]


@dataclass
class _KDNode:
    item_id: int
    point: list[float]
    axis: int
    left: _KDNode | None = None
    right: _KDNode | None = None


class KDTree(BruteForce):
    """Exact KD-tree with metric-specific, conservative split-plane pruning.

    Cosine search builds a tree of normalized nonzero vectors. For unit vectors,
    squared Euclidean distance / 2 equals cosine distance. Zero vectors are
    handled separately because that identity does not apply to them.
    """

    def __init__(self, items: Mapping[int, Sequence[float]]):
        super().__init__(items)
        self.zero_ids = sorted(i for i, v in self.vectors.items() if not any(v))
        self.root = self._build(list(self.vectors.items()), 0)
        self.cosine_root = self._build(
            [(i, _normalize(v)) for i, v in self.vectors.items() if any(v)], 0
        )

    def _build(self, points: list[tuple[int, list[float]]], depth: int) -> _KDNode | None:
        if not points:
            return None
        axis = depth % self.dimension
        points.sort(key=lambda p: (p[1][axis], p[0]))
        mid = len(points) // 2
        item_id, point = points[mid]
        return _KDNode(item_id, point, axis,
                       self._build(points[:mid], depth + 1),
                       self._build(points[mid + 1:], depth + 1))

    def search(self, query: Sequence[float], k: int = 5, metric: str = "cosine") -> list[tuple[int, float]]:
        q = self._query(query, k, metric)
        if not self.vectors:
            return []
        k = min(k, len(self.vectors))
        if metric == "cosine" and not any(q):
            ranked = sorted((0.0 if not any(v) else 1.0, i) for i, v in self.vectors.items())
            return [(i, d) for d, i in ranked[:k]]
        point = _normalize(q) if metric == "cosine" else q
        root = self.cosine_root if metric == "cosine" else self.root
        best: list[tuple[float, int]] = []

        def offer(item_id: int, value: float) -> None:
            entry = (-value, -item_id)
            if len(best) < k:
                heapq.heappush(best, entry)
            elif entry > best[0]:
                heapq.heapreplace(best, entry)

        if metric == "cosine":
            for item_id in self.zero_ids:
                offer(item_id, 1.0)

        def visit(node: _KDNode | None) -> None:
            if node is None:
                return
            offer(node.item_id, distance(q, self.vectors[node.item_id], metric))
            delta = point[node.axis] - node.point[node.axis]
            near, far = (node.left, node.right) if delta <= 0 else (node.right, node.left)
            visit(near)
            bound = delta * delta / 2.0 if metric == "cosine" else abs(delta)
            worst = -best[0][0] if len(best) == k else math.inf
            # Equality must traverse the opposite side to preserve ID tie order.
            if bound <= worst + 1e-12:
                visit(far)

        visit(root)
        return [(i, d) for d, i in sorted((-d, -i) for d, i in best)]


class _HNSWGraph:
    def __init__(self, vectors: dict[int, list[float]], metric: str,
                 m: int, ef_construction: int, seed: int):
        self.vectors, self.metric, self.m = vectors, metric, m
        self.ef_construction = ef_construction
        self.layers: list[dict[int, set[int]]] = []
        self.levels: dict[int, int] = {}
        self.entry: int | None = None
        self.rng = random.Random(seed)
        for item_id in sorted(vectors):
            self._insert(item_id)

    def _distance(self, q: Sequence[float], item_id: int) -> float:
        return distance(q, self.vectors[item_id], self.metric)

    def _greedy(self, q: Sequence[float], entry: int, layer: int) -> int:
        current = (self._distance(q, entry), entry)
        while True:
            candidate = min([current] + [(self._distance(q, i), i)
                                       for i in self.layers[layer].get(current[1], ())])
            if candidate >= current:
                return current[1]
            current = candidate

    def _beam(self, q: Sequence[float], entries: list[int], layer: int, ef: int) -> list[tuple[float, int]]:
        candidates: list[tuple[float, int]] = []
        best: list[tuple[float, int]] = []
        visited = set(entries)
        for item_id in entries:
            d = self._distance(q, item_id)
            heapq.heappush(candidates, (d, item_id))
            heapq.heappush(best, (-d, -item_id))
        while candidates:
            d, item_id = heapq.heappop(candidates)
            if len(best) >= ef and (d, item_id) > (-best[0][0], -best[0][1]):
                break
            for neighbor in sorted(self.layers[layer].get(item_id, ())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = self._distance(q, neighbor)
                entry = (-nd, -neighbor)
                if len(best) < ef or entry > best[0]:
                    heapq.heappush(candidates, (nd, neighbor))
                    heapq.heappush(best, entry)
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, -i) for d, i in best)

    def _insert(self, item_id: int) -> None:
        level = 0
        while level < 16 and self.rng.random() < 1.0 / self.m:
            level += 1
        self.levels[item_id] = level
        old_top = len(self.layers) - 1
        while len(self.layers) <= level:
            self.layers.append({})
        for layer in range(level + 1):
            self.layers[layer][item_id] = set()
        if self.entry is None:
            self.entry = item_id
            return
        q = self.vectors[item_id]
        entry = self.entry
        for layer in range(old_top, level, -1):
            entry = self._greedy(q, entry, layer)
        for layer in range(min(old_top, level), -1, -1):
            candidates = self._beam(q, [entry], layer, self.ef_construction)
            limit = 2 * self.m if layer == 0 else self.m
            neighbors = [i for _, i in candidates[:limit]]
            self.layers[layer][item_id].update(neighbors)
            for neighbor in neighbors:
                links = self.layers[layer][neighbor]
                links.add(item_id)
                if len(links) > limit:
                    keep = sorted(links, key=lambda i: (self._distance(self.vectors[neighbor], i), i))[:limit]
                    self.layers[layer][neighbor] = set(keep)
            if candidates:
                entry = candidates[0][1]
        if level > old_top:
            self.entry = item_id

    def search(self, q: list[float], k: int, ef: int) -> list[tuple[int, float]]:
        if self.entry is None:
            return []
        entry = self.entry
        for layer in range(len(self.layers) - 1, 0, -1):
            entry = self._greedy(q, entry, layer)
        ranked = self._beam(q, [entry], 0, max(k, ef))
        return [(i, d) for d, i in ranked[:k]]


class HNSW(BruteForce):
    """Small, real multilayer HNSW implementation for learning and experiments.

    A separate graph is constructed lazily for each requested metric. Graph
    construction is excluded from VectorLab's warmed benchmark timings.
    Deletion in VectorLab rebuilds indexes; tombstones are not implemented.
    """

    def __init__(self, items: Mapping[int, Sequence[float]], m: int = 8,
                 ef_construction: int = 64, ef_search: int = 32, seed: int = 42):
        super().__init__(items)
        for name, value in (("m", m), ("ef_construction", ef_construction), ("ef_search", ef_search)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if m < 2 or ef_construction < m:
            raise ValueError("m must be at least 2 and ef_construction must be at least m.")
        self.m, self.ef_construction, self.ef_search, self.seed = m, ef_construction, ef_search, seed
        self._graphs: dict[str, _HNSWGraph] = {}

    def prepare(self, metric: str = "cosine") -> None:
        _metric(metric)
        if metric not in self._graphs:
            self._graphs[metric] = _HNSWGraph(self.vectors, metric, self.m, self.ef_construction, self.seed)

    def search(self, query: Sequence[float], k: int = 5, metric: str = "cosine") -> list[tuple[int, float]]:
        q = self._query(query, k, metric)
        self.prepare(metric)
        return self._graphs[metric].search(q, min(k, len(self.vectors)), self.ef_search)

    def info(self, metric: str = "cosine") -> dict:
        self.prepare(metric)
        graph = self._graphs[metric]
        layers = [{"level": level, "nodes": len(nodes),
                   "directed_edges": sum(len(links) for links in nodes.values())}
                  for level, nodes in enumerate(graph.layers)]
        return {"nodes": len(self.vectors), "dimensions": self.dimension,
                "layers": layers, "layer_count": len(layers),
                "entry_id": graph.entry, "m": self.m,
                "ef_construction": self.ef_construction, "ef_search": self.ef_search,
                "metric": metric, "approximate": True}


_DEMO_LABELS = {
    "CS": ["binary tree", "graph algorithm", "database", "Python programming", "machine learning",
           "neural network", "computer network", "operating system", "sorting algorithm", "web development"],
    "Math": ["calculus", "linear algebra", "probability", "geometry", "statistics",
             "matrix multiplication", "differential equation", "number theory", "trigonometry", "optimization"],
    "Food": ["pizza", "sushi", "pasta", "biryani", "chocolate cake",
             "vegetable salad", "ice cream", "fruit smoothie", "sandwich", "noodle soup"],
    "Sports": ["basketball", "football", "cricket", "tennis", "swimming",
               "badminton", "running", "volleyball", "hockey", "cycling"],
}


def demo_items() -> list[dict]:
    """Forty fixed-seed 16D teaching vectors with four deliberately planted groups."""
    rng = random.Random(2026)
    result = []
    for category_index, (category, labels) in enumerate(_DEMO_LABELS.items()):
        for position, label in enumerate(labels):
            vector = [rng.uniform(0.01, 0.16) for _ in range(16)]
            for axis in range(category_index * 4, category_index * 4 + 4):
                vector[axis] = 0.72 + rng.uniform(-0.12, 0.22)
            vector[category_index * 4 + position % 4] += 0.18
            result.append({"id": len(result) + 1, "label": label, "category": category,
                           "vector": [round(x, 6) for x in vector]})
    return result


def pca_project(vectors: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Deterministic centered PCA via covariance and orthogonal power iteration.

    No external numeric library is needed. This teaching implementation targets
    the small, 16D playground; it is not intended for large embedding matrices.
    """
    if not vectors:
        return []
    rows = [validate_vector(v, len(vectors[0])) for v in vectors]
    n, d = len(rows), len(rows[0])
    means = [math.fsum(v[j] for v in rows) / n for j in range(d)]
    centered = [[v[j] - means[j] for j in range(d)] for v in rows]
    covariance = [[math.fsum(v[i] * v[j] for v in centered) / max(1, n - 1)
                   for j in range(d)] for i in range(d)]
    components: list[list[float]] = []
    for component in range(min(2, d)):
        rng = random.Random(1103 + component)
        direction = _normalize([rng.uniform(-1, 1) for _ in range(d)])
        for _ in range(180):
            candidate = [math.fsum(covariance[i][j] * direction[j] for j in range(d)) for i in range(d)]
            for previous in components:
                overlap = math.fsum(a * b for a, b in zip(candidate, previous))
                candidate = [a - overlap * b for a, b in zip(candidate, previous)]
            if math.hypot(*candidate) < 1e-14:
                direction = [0.0] * d
                break
            candidate = _normalize(candidate)
            change = min(math.dist(candidate, direction), math.dist(candidate, [-x for x in direction]))
            direction = candidate
            if change < 1e-11:
                break
        anchor = max(range(d), key=lambda i: abs(direction[i]))
        if direction[anchor] < 0:
            direction = [-x for x in direction]
        components.append(direction)
    while len(components) < 2:
        components.append([0.0] * d)
    return [(math.fsum(x * y for x, y in zip(v, components[0])),
             math.fsum(x * y for x, y in zip(v, components[1]))) for v in centered]


class VectorLab:
    """Mutable, in-memory demo workspace; document persistence is a separate concern."""

    def __init__(self, items: list[dict] | None = None):
        source = demo_items() if items is None else items
        self.items: list[dict] = []
        self.dimension = len(source[0]["vector"]) if source else 16
        seen = set()
        for item in source:
            item_id = item["id"]
            if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id in seen:
                raise ValueError("Each demo item needs a unique integer ID.")
            seen.add(item_id)
            label, category = self._labels(item["label"], item["category"])
            self.items.append({"id": item_id, "label": label, "category": category,
                               "vector": validate_vector(item["vector"], self.dimension)})
        self._next_id = max(seen, default=0) + 1
        self._rebuild()

    @staticmethod
    def _labels(label: str, category: str) -> tuple[str, str]:
        if not isinstance(label, str) or not label.strip() or len(label.strip()) > 100:
            raise ValueError("Label must be between 1 and 100 characters.")
        if not isinstance(category, str) or not category.strip() or len(category.strip()) > 40:
            raise ValueError("Category must be between 1 and 40 characters.")
        return label.strip(), category.strip()

    def _rebuild(self) -> None:
        vectors = {item["id"]: item["vector"] for item in self.items}
        self._indexes = {"brute_force": BruteForce(vectors), "kd_tree": KDTree(vectors), "hnsw": HNSW(vectors)}
        self._by_id = {item["id"]: item for item in self.items}
        for metric in METRICS:
            self._indexes["hnsw"].prepare(metric)

    def _resolve_query(self, query: str | list[float]) -> list[float]:
        if not isinstance(query, str):
            return validate_vector(query, self.dimension)
        text = query.strip().casefold()
        if not text:
            raise ValueError("Enter a demo concept, such as binary tree, calculus, pizza, or cricket.")
        for item in self.items:
            if item["label"].casefold() == text:
                return list(item["vector"])
        tokens = set(re.findall(r"\w+", text))
        scored = []
        for item in self.items:
            label_tokens = set(re.findall(r"\w+", item["label"].casefold()))
            overlap = tokens & label_tokens
            if overlap:
                scored.append((len(overlap) / len(tokens | label_tokens), -item["id"], item))
        if scored:
            return list(max(scored, key=lambda row: (row[0], row[1]))[2]["vector"])
        category_items = [item for item in self.items if item["category"].casefold() == text]
        if category_items:
            return [math.fsum(item["vector"][j] for item in category_items) / len(category_items)
                    for j in range(self.dimension)]
        raise ValueError("No demo label matched. Try binary tree, calculus, pizza, cricket, or a category "
                         "(CS, Math, Food, Sports). This playground uses synthetic demo vectors, "
                         "not a text embedding model; use Documents for your own content.")

    @staticmethod
    def _algorithm(algorithm: str) -> str:
        aliases = {"kdtree": "kd_tree", "kd-tree": "kd_tree", "brute": "brute_force",
                   "bruteforce": "brute_force", "brute-force": "brute_force"}
        if not isinstance(algorithm, str):
            raise ValueError("Algorithm must be hnsw, kd_tree, or brute_force.")
        name = aliases.get(algorithm.casefold(), algorithm.casefold())
        if name not in ALGORITHMS:
            raise ValueError("Algorithm must be hnsw, kd_tree, or brute_force.")
        return name

    def _results(self, matches: list[tuple[int, float]]) -> list[dict]:
        return [{"id": i, "label": self._by_id[i]["label"], "category": self._by_id[i]["category"],
                 "distance": d} for i, d in matches]

    def search(self, query: str | list[float], k: int = 5, metric: str = "cosine",
               algorithm: str = "hnsw") -> dict:
        q, algorithm = self._resolve_query(query), self._algorithm(algorithm)
        _k(k)
        _metric(metric)
        start = time.perf_counter()
        matches = self._indexes[algorithm].search(q, k, metric)
        elapsed = (time.perf_counter() - start) * 1000
        return {"results": self._results(matches), "elapsed_ms": elapsed,
                "query_vector": q, "algorithm": algorithm, "metric": metric,
                "query_mode": "synthetic demo label lookup" if isinstance(query, str) else "numeric vector",
                "approximate": algorithm == "hnsw"}

    def benchmark(self, query: str | list[float], k: int = 5, metric: str = "cosine", repeats: int = 5) -> list[dict]:
        q = self._resolve_query(query)
        _k(k)
        _metric(metric)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 50:
            raise ValueError("repeats must be an integer between 1 and 50.")
        truth = {i for i, _ in self._indexes["brute_force"].search(q, k, metric)}
        rows = []
        for algorithm in ALGORITHMS:
            index = self._indexes[algorithm]
            index.search(q, k, metric)  # warm caches; never time graph construction
            times = []
            matches = []
            for _ in range(repeats):
                start = time.perf_counter()
                matches = index.search(q, k, metric)
                times.append((time.perf_counter() - start) * 1000)
            recall = len(truth & {i for i, _ in matches}) / len(truth) if truth else 1.0
            rows.append({"algorithm": algorithm, "mean_ms": statistics.mean(times),
                         "recall_at_k": recall, "results": self._results(matches),
                         "repeats": repeats, "approximate": algorithm == "hnsw"})
        return rows

    def add(self, label: str, category: str, vector: list[float]) -> int:
        label, category = self._labels(label, category)
        vector = validate_vector(vector, self.dimension)
        if any(item["label"].casefold() == label.casefold() for item in self.items):
            raise ValueError("That label already exists. Choose a unique label.")
        item_id = self._next_id
        self._next_id += 1
        self.items.append({"id": item_id, "label": label, "category": category, "vector": vector})
        self._rebuild()
        return item_id

    def delete(self, item_id: int) -> None:
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id not in self._by_id:
            raise ValueError("Demo vector ID was not found.")
        self.items = [item for item in self.items if item["id"] != item_id]
        self._rebuild()

    def project(self) -> list[dict]:
        coordinates = pca_project([item["vector"] for item in self.items])
        return [{"id": item["id"], "label": item["label"], "category": item["category"], "x": x, "y": y}
                for item, (x, y) in zip(self.items, coordinates)]

    def info(self) -> dict:
        graph = self._indexes["hnsw"].info()
        return {**graph, "dimensions": self.dimension, "count": len(self.items),
                "categories": sorted({item["category"] for item in self.items}),
                "algorithms": list(ALGORITHMS), "metrics": list(METRICS), "hnsw": graph,
                "persistence": "in memory; resets when the server restarts",
                "vector_source": "synthetic hand-designed category vectors; not learned embeddings",
                "timing_note": "Warmed retrieval only; small datasets may favor brute force. "
                               "HNSW recall is approximate and no logarithmic runtime is guaranteed."}
