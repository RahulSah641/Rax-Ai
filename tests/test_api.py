"""Real HTTP integration and local API boundary tests; no network services needed."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from your_ai.api import MAX_BODY_BYTES, create_server
from your_ai.knowledge import KnowledgeStore
from your_ai.vectors import VectorLab


class LocalAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = KnowledgeStore(Path(self.temp.name) / "knowledge.sqlite3")
        self.lab = VectorLab([
            {"id": 1, "label": "Alpha", "category": "Demo", "vector": [1.0, 0.0]},
            {"id": 2, "label": "Beta", "category": "Demo", "vector": [0.0, 1.0]},
        ])
        self.token = "integration-test-token"
        self.server = create_server(self.store, self.lab, port=0, token=self.token)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None, *, auth=True, headers=None, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        request_headers = {"Authorization": "Bearer " + self.token} if auth else {}
        if body is not None or raw is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        try:
            connection.request(method, path, body=data, headers=request_headers)
            response = connection.getresponse()
            content = response.read()
            return response.status, dict(response.getheaders()), json.loads(content) if content else None
        finally:
            connection.close()

    def test_health_is_public_but_data_needs_token(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        status, headers, body = self.request("GET", "/health", auth=False)
        self.assertEqual((status, body), (200, {"status": "ok"}))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        for path in ("/api/stats", "/api/documents", "/api/vectors", "/api/unknown"):
            status, headers, body = self.request("GET", path, auth=False)
            self.assertEqual(status, 401, path)
            self.assertIn("WWW-Authenticate", headers)
        self.assertEqual(self.request("GET", "/api/stats",
                         headers={"Authorization": "Bearer wrong"})[0], 401)
        status, _, body = self.request("GET", "/api/stats")
        self.assertEqual(status, 200)
        self.assertEqual(body["knowledge"]["documents"], 0)
        self.assertEqual(body["vectors"]["count"], 2)

    def test_document_crud_search_and_grounded_offline_answer(self):
        status, _, created = self.request("POST", "/api/documents", {
            "title": "Python notes", "text": "Python is used for automation and software development. "
            "SQLite stores structured information in a local database file."
        })
        self.assertEqual(status, 201)
        item_id = created["id"]
        listed = self.request("GET", "/api/documents")[2]["documents"]
        self.assertEqual(listed[0]["id"], item_id)
        self.assertNotIn("text", listed[0])
        status, _, result = self.request("POST", "/api/search", {"query": "Python automation", "k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(result["results"][0]["document_id"], item_id)
        status, _, answer = self.request("POST", "/api/ask", {"question": "What is Python used for?", "k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(answer["mode"], "offline")
        self.assertTrue(answer["sources"])
        self.assertIn("automation", answer["answer"])
        self.assertEqual(self.request("DELETE", "/api/documents/" + item_id)[0], 200)
        self.assertEqual(self.request("DELETE", "/api/documents/" + item_id)[0], 404)
        self.assertEqual(self.request("GET", "/api/documents")[2]["documents"], [])

    def test_vector_crud_all_search_algorithms_and_benchmark(self):
        status, _, created = self.request("POST", "/api/vectors", {
            "label": "Gamma", "category": "Demo", "vector": [0.8, 0.2]
        })
        self.assertEqual(status, 201)
        item_id = created["id"]
        self.assertEqual(len(self.request("GET", "/api/vectors")[2]["vectors"]), 3)
        for algorithm in ("hnsw", "kd_tree", "brute_force"):
            for metric in ("cosine", "euclidean", "manhattan"):
                status, _, result = self.request("POST", "/api/vector-search", {
                    "query": "Gamma", "k": 2, "metric": metric, "algorithm": algorithm
                })
                self.assertEqual(status, 200, (algorithm, metric, result))
                self.assertEqual(result["results"][0]["id"], item_id)
                self.assertEqual(len(result["results"]), 2)
        status, _, rows = self.request("POST", "/api/benchmark", {
            "query": [1, 0], "k": 2, "metric": "cosine", "repeats": 2
        })
        self.assertEqual(status, 200)
        self.assertEqual({row["algorithm"] for row in rows}, {"hnsw", "kd_tree", "brute_force"})
        self.assertTrue(all(0 <= row["recall_at_k"] <= 1 for row in rows))
        self.assertEqual(self.request("DELETE", f"/api/vectors/{item_id}")[0], 200)
        self.assertEqual(self.request("DELETE", f"/api/vectors/{item_id}")[0], 404)
        self.assertEqual(len(self.request("GET", "/api/vectors")[2]["vectors"]), 2)

    def test_strict_types_ranges_modes_and_unknown_fields(self):
        invalid = [
            ("/api/search", {"query": "python", "k": True}),
            ("/api/search", {"query": "python", "k": 2.0}),
            ("/api/search", {"query": "python", "k": 21}),
            ("/api/search", {"query": "python", "mode": "ollama"}),
            ("/api/search", {"query": "   "}),
            ("/api/ask", {"question": "python", "mode": "remote"}),
            ("/api/ask", {"question": "x" * 4001}),
            ("/api/documents", {"title": "Title"}),
            ("/api/documents", {"title": "Title", "text": "content", "file_path": "/tmp/test"}),
            ("/api/documents", {"title": "Title", "text": "x" * 200001}),
            ("/api/vectors", {"label": "Gamma", "category": "Demo", "vector": [True, 0]}),
            ("/api/vectors", {"label": "Gamma", "category": "Demo", "vector": [1, 0, 0]}),
            ("/api/vectors", {"label": "Gamma", "category": "Demo", "vector": [1e308, 0]}),
            ("/api/vector-search", {"query": [1, 0], "algorithm": "unknown"}),
            ("/api/vector-search", {"query": [1, 0], "metric": "unknown"}),
            ("/api/vector-search", {"query": []}),
            ("/api/benchmark", {"query": [1, 0], "repeats": True}),
            ("/api/benchmark", {"query": [1, 0], "repeats": 26}),
        ]
        for path, body in invalid:
            with self.subTest(path=path, body=str(body)[:120]):
                self.assertEqual(self.request("POST", path, body)[0], 400)
        self.assertEqual(self.lab.info()["count"], 2)
        self.assertEqual(self.store.stats()["documents"], 0)

    def test_malformed_json_nonfinite_and_invalid_unicode(self):
        for raw in (b"not json", b"[]", b'{"query":NaN}', b'{"query":Infinity}',
                    b'{"query":[1e400,0]}', b'{"query":"one","query":"two"}',
                    b'{"query":"\\ud800"}', b'{"query":"\xff"}'):
            with self.subTest(raw=raw):
                self.assertEqual(self.request("POST", "/api/vector-search", raw=raw)[0], 400)

    def test_body_content_type_size_and_transfer_encoding(self):
        self.assertEqual(self.request("POST", "/api/search", raw=b"{}",
                         headers={"Content-Type": "text/plain"})[0], 415)
        # A declared oversized body is rejected before any body is read.
        self.assertEqual(self.request("POST", "/api/search", raw=b"",
                         headers={"Content-Length": str(MAX_BODY_BYTES + 1)})[0], 413)
        self.assertEqual(self.request("POST", "/api/search", raw=b"{}",
                         headers={"Transfer-Encoding": "chunked"})[0], 400)

    def test_foreign_origins_hosts_and_cors_preflight_are_rejected(self):
        for origin in ("https://evil.example", "null", "http://localhost:1"):
            self.assertEqual(self.request("GET", "/api/stats", headers={"Origin": origin})[0], 403)
        for host in ("evil.example", "127.0.0.1.evil.example", "127.0.0.1:1", "localhost@evil.example"):
            self.assertEqual(self.request("GET", "/health", headers={"Host": host})[0], 403)
        self.assertEqual(self.request("GET", "/api/stats",
                         headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)
        status, headers, _ = self.request("OPTIONS", "/api/search",
                                          headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("PUT", "/api/search")[0], 405)

    def test_unknown_paths_and_invalid_deletion_ids(self):
        self.assertEqual(self.request("GET", "/api/missing")[0], 404)
        self.assertEqual(self.request("GET", "/api/stats?debug=true")[0], 400)
        self.assertEqual(self.request("DELETE", "/api/documents/../../secret")[0], 400)
        self.assertEqual(self.request("DELETE", "/api/vectors/not-a-number")[0], 400)
        self.assertEqual(self.request("DELETE", "/api/vectors/999999")[0], 404)

    def test_generated_token_and_server_argument_validation(self):
        second = create_server(self.store, self.lab, port=0)
        try:
            self.assertGreaterEqual(len(second.api_token), 32)
            self.assertNotEqual(second.api_token, self.token)
            self.assertEqual(second.server_address[0], "127.0.0.1")
        finally:
            second.server_close()
        for token in ("", "has space", "line\nbreak", 123):
            with self.assertRaises(ValueError):
                create_server(self.store, self.lab, port=0, token=token)
        for port in (True, -1, 65536, "8765"):
            with self.assertRaises(ValueError):
                create_server(self.store, self.lab, port=port)


if __name__ == "__main__":
    unittest.main()
