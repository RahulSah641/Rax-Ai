import io
import json
import math
import unittest
import urllib.error
from unittest.mock import patch

from your_ai.ollama_client import OllamaClient, OllamaError, validate_vector


class OllamaTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def response(self, payload):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    def test_endpoint_only_loopback_no_credentials_or_redirect_path(self):
        for url in ["http://example.com:11434", "http://192.168.1.2:11434", "http://0.0.0.0:11434",
                    "file:///etc/passwd", "http://user:pass@localhost:11434", "http://localhost:11434/api",
                    "http://localhost:11434?cloud=1", "http://localhost:11434#fragment", "http://localhost:bad"]:
            with self.subTest(url=url), self.assertRaises(OllamaError):
                OllamaClient(base_url=url)
        self.assertEqual(OllamaClient(base_url="http://localhost:11434/").base_url, "http://127.0.0.1:11434")
        self.assertEqual(OllamaClient(base_url="http://[::1]:11434").base_url, "http://[::1]:11434")

    def test_model_names_and_timeout_validated(self):
        for model in ["", "model name", "model;rm -rf", "$(whoami)"]:
            with self.assertRaises(OllamaError):
                OllamaClient(embedding_model=model)
        for timeout in [0, 601, True, float("nan")]:
            with self.assertRaises(OllamaError):
                OllamaClient(timeout=timeout)

    def test_status_online_and_offline(self):
        with patch.object(self.client._opener, "open", return_value=self.response({"models": [{"name": "nomic-embed-text:latest"}]})):
            self.assertEqual(self.client.status()["models"], ["nomic-embed-text:latest"])
        with patch.object(self.client._opener, "open", side_effect=urllib.error.URLError("connection refused")):
            status = self.client.status()
            self.assertFalse(status["online"])
            self.assertIn("Offline mode", status["message"])

    def test_embed_uses_modern_api_and_normalizes_vectors(self):
        with patch.object(self.client._opener, "open", return_value=self.response({"embeddings": [[3, 4], [0, 2]]})) as call:
            result = self.client.embed(["first", "second"])
            request = call.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embed")
        self.assertEqual(json.loads(request.data), {"model": "nomic-embed-text", "input": ["first", "second"], "truncate": False})
        self.assertEqual(result, [[.6, .8], [0, 1]])

    def test_embed_rejects_shape_and_nonfinite_data(self):
        for payload in [{"embeddings": []}, {"embeddings": [[0, 0]]}, {"embeddings": [[float("nan")]]},
                        {"embeddings": [[True, False]]}, {"embeddings": [["number"]]}]:
            with patch.object(self.client._opener, "open", return_value=self.response(payload)), self.assertRaises(OllamaError):
                self.client.embed(["one"])
        with patch.object(self.client._opener, "open", return_value=self.response({"embeddings": [[1, 0], [1]]})), self.assertRaisesRegex(OllamaError, "inconsistent"):
            self.client.embed(["one", "two"])

    def test_embed_input_validation(self):
        for texts in [[], [""], ["x"] * 33, "not a list", [None]]:
            with self.assertRaises(OllamaError):
                self.client.embed(texts)

    def test_http_error_explains_model_installation(self):
        error = urllib.error.HTTPError("http://localhost/api/embed", 404, "Missing", {}, io.BytesIO(b'{"error":"model not found"}'))
        with patch.object(self.client._opener, "open", side_effect=error), self.assertRaisesRegex(OllamaError, "selected model is installed"):
            self.client.embed(["document"])

    def test_http_redirect_is_disabled(self):
        handler = next(handler for handler in self.client._opener.handlers if handler.__class__.__name__ == "_NoRedirect")
        self.assertIsNone(handler.redirect_request(None, None, 302, "redirect", {}, "https://external.example"))

    def test_invalid_json_and_oversized_responses(self):
        with patch.object(self.client._opener, "open", return_value=io.BytesIO(b"not json")), self.assertRaisesRegex(OllamaError, "invalid JSON"):
            self.client.embed(["one"])
        with patch("your_ai.ollama_client.MAX_RESPONSE_BYTES", 32), patch.object(self.client._opener, "open", return_value=io.BytesIO(b"x" * 40)), self.assertRaisesRegex(OllamaError, "safety limit"):
            self.client.embed(["one"])
        with patch.object(self.client._opener, "open", return_value=self.response(["wrong root"])), self.assertRaisesRegex(OllamaError, "unexpected JSON"):
            self.client.embed(["one"])

    def test_model_digest_invalidates_embedding_cache_identity(self):
        for digest in ["abc", "def"]:
            with patch.object(self.client, "_model_details", return_value=[{"name": "nomic-embed-text:latest", "digest": digest}]):
                self.assertTrue(self.client.embedding_cache_key().endswith("|" + digest))
        with patch.object(self.client, "_model_details", return_value=[{"name": "nomic-embed-text:latest"}]):
            self.assertNotEqual(self.client.embedding_cache_key(), self.client.embedding_cache_key())

    def test_missing_model_actionable(self):
        with patch.object(self.client, "_model_details", return_value=[]), self.assertRaisesRegex(OllamaError, "ollama pull nomic-embed-text"):
            self.client.embedding_cache_key()

    def test_generate_has_grounded_prompt_and_source_labels(self):
        with patch.object(self.client, "_require_model"), patch.object(self.client._opener, "open", return_value=self.response({"response": "Answer [1]"})) as call:
            answer = self.client.generate("What is solar?", [{"title": "Energy", "text": "Solar is sunlight."}])
            request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertFalse(payload["stream"])
        self.assertIn("untrusted", payload["system"])
        self.assertIn("[1] Energy", payload["prompt"])
        self.assertEqual(answer, "Answer [1]")

    def test_generate_never_reports_empty_success(self):
        with patch.object(self.client, "_require_model"), patch.object(self.client._opener, "open", return_value=self.response({"response": ""})), self.assertRaisesRegex(OllamaError, "no answer"):
            self.client.generate("question", [{"title": "A", "text": "B"}])
        with self.assertRaises(OllamaError):
            self.client.generate("question", [])

    def test_vector_normalization_avoids_overflow(self):
        vector = validate_vector([1e200, 1e200])
        self.assertAlmostEqual(sum(x*x for x in vector), 1)
        with self.assertRaises(OllamaError):
            validate_vector([float("inf")])


if __name__ == "__main__":
    unittest.main()
