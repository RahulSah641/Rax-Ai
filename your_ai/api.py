"""Authenticated, loopback-only JSON API using Python's standard library.

Run through ``python app.py --api``. This is a local development interface,
not a production internet-facing server. The desktop UI does not need it.
"""

from __future__ import annotations

import hmac
import copy
import json
import math
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


MAX_BODY_BYTES = 1024 * 1024
MAX_K = 20
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class APIError(Exception):
    """An error that is safe to return to the local API caller."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise APIError(400, f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _string(value, name, max_length=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise APIError(400, f"{name} must be non-empty text, at most {max_length} characters.")
    return value


def _finite_tree(value, depth=0):
    if depth > 30:
        raise APIError(400, "JSON nesting is too deep.")
    if isinstance(value, float) and not math.isfinite(value):
        raise APIError(400, "JSON numbers must be finite.")
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise APIError(400, "JSON text must contain valid Unicode characters.") from None
    if isinstance(value, dict):
        for item in (*value.keys(), *value.values()):
            _finite_tree(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item, depth + 1)


def _reject_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property.")
        result[key] = value
    return result


def _fields(body, required, optional=()):
    missing = set(required) - body.keys()
    unknown = body.keys() - set(required) - set(optional)
    if missing:
        raise APIError(400, "Missing field(s): " + ", ".join(sorted(missing)))
    if unknown:
        raise APIError(400, "Unknown field(s): " + ", ".join(sorted(unknown)))


def _vector(value, name="vector"):
    if not isinstance(value, list) or not 1 <= len(value) <= 4096:
        raise APIError(400, f"{name} must be a list of 1 to 4096 finite numbers.")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise APIError(400, f"{name} must contain finite numbers, without booleans.")
    return value


def _query(value):
    return _string(value, "query") if isinstance(value, str) else _vector(value, "query")


class LocalAPIServer(ThreadingHTTPServer):
    """Server shared state; all vector operations use one re-entrant lock."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, store, lab, port, token):
        self.store = store
        self.lab = lab
        self.api_token = token
        self.lab_lock = threading.RLock()
        super().__init__(("127.0.0.1", port), APIHandler)


class APIHandler(BaseHTTPRequestHandler):
    server_version = "Rax Ai/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, fmt, *args):
        # Avoid logging questions, document data, or authentication tokens.
        return

    def _send_json(self, status, payload):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if status == 401:
            self.send_header("WWW-Authenticate", 'Bearer realm="Rax Ai"')
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _guard(self):
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:
            raise APIError(400, "Exactly one Host header is required.")
        host = hosts[0]
        if host != host.strip() or any(char in host for char in "/\\@?# \t\r\n"):
            raise APIError(403, "Only a loopback Host is allowed.")
        try:
            parsed_host = urlsplit("//" + host)
            hostname = parsed_host.hostname
            host_port = parsed_host.port
        except ValueError:
            raise APIError(403, "Only a loopback Host is allowed.") from None
        port = self.server.server_address[1]
        if hostname not in ("127.0.0.1", "localhost") or host_port not in (None, port):
            raise APIError(403, "Only this server's loopback Host is allowed.")
        origins = self.headers.get_all("Origin", [])
        if origins:
            allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if len(origins) != 1 or origins[0] not in allowed:
                raise APIError(403, "Foreign browser origins are not allowed.")
        # Token is mandatory for every API request, including unknown routes.
        if self.path.startswith("/api/"):
            auth = self.headers.get_all("Authorization", [])
            expected = "Bearer " + self.server.api_token
            if len(auth) != 1 or not hmac.compare_digest(
                auth[0].encode("utf-8"), expected.encode("utf-8")
            ):
                raise APIError(401, "A valid Authorization: Bearer token is required.")

    def _read_json(self):
        if self.headers.get("Transfer-Encoding"):
            raise APIError(400, "Transfer-Encoding is not supported. Use Content-Length.")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise APIError(411, "Content-Length is required.")
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            raise APIError(400, "Invalid Content-Length.")
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise APIError(413, "Request body must not exceed 1 MiB.")
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise APIError(415, "Content-Type must be application/json.")
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise APIError(400, "Incomplete request body.")
            body = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                              object_pairs_hook=_unique_object)
        except (ValueError, UnicodeError, RecursionError):
            raise APIError(400, "Body must be valid UTF-8 JSON without duplicate keys or non-finite numbers.") from None
        if not isinstance(body, dict):
            raise APIError(400, "JSON body must be an object.")
        _finite_tree(body)
        return body

    def _route(self):
        path = self.path
        if "?" in path or "#" in path or not path.startswith("/"):
            raise APIError(400, "Use an exact endpoint path without query parameters.")
        store, lab = self.server.store, self.server.lab
        if self.command == "GET":
            if path == "/health":
                return 200, {"status": "ok"}
            if path == "/api/stats":
                with self.server.lab_lock:
                    return 200, {"knowledge": store.stats(), "vectors": lab.info()}
            if path == "/api/documents":
                return 200, {"documents": store.list_documents()}
            if path == "/api/vectors":
                with self.server.lab_lock:
                    return 200, {"vectors": copy.deepcopy(lab.items)}
        elif self.command == "DELETE":
            for prefix, target in (("/api/documents/", "document"), ("/api/vectors/", "vector")):
                if path.startswith(prefix):
                    item_id = path[len(prefix):]
                    if not _ID_RE.fullmatch(item_id):
                        raise APIError(400, "Invalid item ID.")
                    if target == "document":
                        removed = store.delete_document(item_id)
                    else:
                        if not re.fullmatch(r"[0-9]{1,18}", item_id):
                            raise APIError(400, "Vector ID must be a non-negative integer.")
                        with self.server.lab_lock:
                            removed = any(item["id"] == int(item_id) for item in lab.items)
                            if removed:
                                lab.delete(int(item_id))
                    if not removed:
                        raise APIError(404, f"The {target} was not found.")
                    return 200, {"deleted": item_id}
        elif self.command == "POST":
            if path not in ("/api/documents", "/api/search", "/api/ask", "/api/vectors",
                            "/api/vector-search", "/api/benchmark"):
                raise APIError(404, "Endpoint not found.")
            body = self._read_json()
            if path == "/api/documents":
                _fields(body, ("title", "text"))
                item_id = store.add_document(_string(body["title"], "title", 240),
                                             _string(body["text"], "text", 200000))
                return 201, {"id": item_id}
            if path in ("/api/search", "/api/ask"):
                field = "query" if path == "/api/search" else "question"
                _fields(body, (field,), ("k", "mode"))
                query = _string(body[field], field)
                k = _integer(body.get("k", 5), "k", 1, MAX_K)
                if body.get("mode", "offline") != "offline":
                    raise APIError(400, "The local REST API supports mode='offline' only.")
                if path == "/api/search":
                    return 200, {"results": store.search(query, k=k, mode="offline")}
                return 200, store.ask(query, k=k, mode="offline")
            if path == "/api/vectors":
                _fields(body, ("label", "category", "vector"))
                label = _string(body["label"], "label", 100)
                category = _string(body["category"], "category", 40)
                vector = _vector(body["vector"])
                with self.server.lab_lock:
                    item_id = lab.add(label=label, category=category, vector=vector)
                return 201, {"id": item_id}
            if path in ("/api/vector-search", "/api/benchmark"):
                optional = ("k", "metric", "algorithm") if path == "/api/vector-search" else ("k", "metric", "repeats")
                _fields(body, ("query",), optional)
                query = _query(body["query"])
                k = _integer(body.get("k", 5), "k", 1, MAX_K)
                metric = body.get("metric", "cosine")
                if metric not in ("cosine", "euclidean", "manhattan"):
                    raise APIError(400, "metric must be cosine, euclidean, or manhattan.")
                if path == "/api/vector-search":
                    algorithm = body.get("algorithm", "hnsw")
                    if algorithm not in ("hnsw", "kd_tree", "brute_force"):
                        raise APIError(400, "algorithm must be hnsw, kd_tree, or brute_force.")
                    with self.server.lab_lock:
                        result = lab.search(query, k=k, metric=metric, algorithm=algorithm)
                    return 200, result
                repeats = _integer(body.get("repeats", 5), "repeats", 1, 25)
                with self.server.lab_lock:
                    result = lab.benchmark(query, k=k, metric=metric, repeats=repeats)
                return 200, result
        elif self.command not in ("GET", "POST", "DELETE"):
            raise APIError(405, "Supported methods are GET, POST, and DELETE.")
        raise APIError(404, "Endpoint not found.")

    def _handle(self):
        try:
            self._guard()
            status, payload = self._route()
            self._send_json(status, payload)
        except APIError as exc:
            self._send_json(exc.status, {"error": str(exc)})
        except (socket.timeout, TimeoutError):
            try:
                self._send_json(408, {"error": "Request timed out."})
            except (OSError, ValueError):
                pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        except KeyError:
            self._send_json(404, {"error": "Requested item was not found."})
        except (ValueError, TypeError, OverflowError) as exc:
            self._send_json(400, {"error": str(exc) or "Invalid request data."})
        except Exception:
            self._send_json(500, {"error": "The operation could not be completed. Check the local application data."})

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_HEAD = _handle


def create_server(store, lab, port=8765, token=None):
    """Create a loopback server. ``port=0`` selects a free port for tests.

    The returned server exposes ``api_token``. When token is omitted, a fresh
    token is generated. ``serve()`` prints it once before accepting requests.
    """
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 to 65535")
    if token is None:
        token = secrets.token_urlsafe(24)
    if not isinstance(token, str) or not 1 <= len(token) <= 512 or any(
        ord(char) < 33 or ord(char) > 126 for char in token
    ):
        raise ValueError("API token must contain 1 to 512 printable ASCII characters without spaces.")
    return LocalAPIServer(store, lab, port, token)


def serve(store, lab, port=8765, token=None):
    """Run the local API until Ctrl+C, closing its socket on exit."""
    server = create_server(store, lab, port=port, token=token)
    print(f"Rax Ai API: http://127.0.0.1:{server.server_address[1]}", flush=True)
    print(f"Bearer token: {server.api_token}", flush=True)
    print("Local computer only. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nAPI stopped.", flush=True)
    finally:
        server.server_close()
