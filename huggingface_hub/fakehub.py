"""Minimal fake Hub for download experiments (HEAD/GET /{repo}/resolve/{rev}/{file}, repo_info, tree listing).

Only the HTTP endpoint is stubbed; huggingface_hub runs unpatched against it via `endpoint=`.
Each file: content bytes, etag (sha256 for LFS-like, 40-hex for regular), commit hash.
GETs are streamed in 64 KiB chunks over `get_seconds` seconds (to keep downloads in flight) and counted.
HEADs answer after `head_seconds` (the real Hub HEAD /resolve latency measured from here: 0.11-0.29 s).
"""

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeHub:
    def __init__(self, files, commit, get_seconds=0.0, head_seconds=0.0):
        self.files = files  # {filename: (content, etag)}
        self.commit = commit
        self.get_seconds = get_seconds
        self.head_seconds = head_seconds  # latency of HEAD /resolve (real Hub: ~0.1-0.3 s)
        self.get_seconds_by_file = {}  # optional per-file override of get_seconds
        self.gets = []  # (time, filename, range)
        self.heads = 0
        self.api_calls = 0
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _lookup(self):
                # /{ns}/{name}/resolve/{rev}/{filename}
                parts = self.path.split("?")[0].lstrip("/").split("/")
                if len(parts) < 5 or parts[2] != "resolve":
                    return None
                return "/".join(parts[4:])

            def _headers(self, content, etag):
                self.send_header("X-Repo-Commit", hub.commit)
                self.send_header("ETag", f'"{etag}"')
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Accept-Ranges", "bytes")

            def do_HEAD(self):
                name = self._lookup()
                if name not in hub.files:
                    self.send_response(404)
                    self.send_header("X-Error-Code", "EntryNotFound")
                    self.send_header("X-Repo-Commit", hub.commit)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                content, etag = hub.files[name]
                with hub.lock:
                    hub.heads += 1
                if hub.head_seconds:
                    time.sleep(hub.head_seconds)
                self.send_response(200)
                self._headers(content, etag)
                self.end_headers()

            def _api(self):
                # /api/models/{ns}/{name}/revision/{rev}  and  /api/models/{ns}/{name}/tree/{rev}
                parts = self.path.split("?")[0].lstrip("/").split("/")
                if len(parts) >= 6 and parts[0] == "api" and parts[4] in ("revision", "tree"):
                    if parts[4] == "revision":
                        body = {"id": f"{parts[2]}/{parts[3]}", "sha": hub.commit, "private": False,
                                "siblings": [{"rfilename": n} for n in hub.files]}
                    else:
                        body = []
                        for n, (content, etag) in hub.files.items():
                            entry = {"type": "file", "path": n, "size": len(content), "oid": hashlib.sha1(n.encode()).hexdigest()}
                            if len(etag) == 64:
                                entry["lfs"] = {"oid": etag, "size": len(content), "pointerSize": 134}
                                entry["size"] = len(content)
                            else:
                                entry["oid"] = etag
                            body.append(entry)
                    data = json.dumps(body).encode()
                    with hub.lock:
                        hub.api_calls += 1
                    if hub.head_seconds:
                        time.sleep(hub.head_seconds)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return True
                return False

            def do_GET(self):
                if self._api():
                    return
                name = self._lookup()
                if name not in hub.files:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                content, etag = hub.files[name]
                with hub.lock:
                    hub.gets.append((time.time(), name, self.headers.get("Range")))
                self.send_response(200)
                self._headers(content, etag)
                self.end_headers()
                chunk = 64 * 1024
                n_chunks = max(1, (len(content) + chunk - 1) // chunk)
                delay = hub.get_seconds_by_file.get(name, hub.get_seconds) / n_chunks
                try:
                    for i in range(0, len(content), chunk):
                        self.wfile.write(content[i : i + chunk])
                        if delay:
                            time.sleep(delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def sha256(b):
    return hashlib.sha256(b).hexdigest()
