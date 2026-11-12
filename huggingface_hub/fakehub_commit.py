"""Minimal fake Hub for commit experiments: create_repo, preupload (always "regular"), commit (ndjson), repo_info.

Only the HTTP endpoint is stubbed; huggingface_hub runs unpatched against it via `HfApi(endpoint=...)`.
`hub.remote[path]` holds the committed content of each file, `hub.commits` the list of committed path lists.
"""

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeCommitHub:
    def __init__(self):
        self.remote = {}
        self.commits = []
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _reply(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                parts = self.path.split("?")[0].lstrip("/").split("/")
                if parts[:3] == ["api", "repos", "create"]:
                    req = json.loads(body)
                    name = f"{req.get('organization') or 'user'}/{req['name']}"
                    prefix = "" if req.get("type") in (None, "model") else f"{req['type']}s/"
                    return self._reply({"url": f"{hub.endpoint}/{prefix}{name}", "name": name})
                if len(parts) >= 6 and parts[4] == "preupload":
                    files = json.loads(body)["files"]
                    with hub.lock:
                        out = [
                            {
                                "path": f["path"],
                                "uploadMode": "regular",
                                "shouldIgnore": False,
                                "oid": git_sha1(hub.remote[f["path"]]) if f["path"] in hub.remote else None,
                            }
                            for f in files
                        ]
                    return self._reply({"files": out})
                if len(parts) >= 6 and parts[4] == "commit":
                    paths = []
                    with hub.lock:
                        for line in body.splitlines():
                            item = json.loads(line)
                            if item["key"] == "file":
                                v = item["value"]
                                hub.remote[v["path"]] = base64.b64decode(v["content"])
                                paths.append(v["path"])
                        hub.commits.append(paths)
                        oid = hashlib.sha1(str(len(hub.commits)).encode()).hexdigest()
                    prefix = "" if parts[1] == "models" else f"{parts[1]}/"
                    url = f"{hub.endpoint}/{prefix}{parts[2]}/{parts[3]}/commit/{oid}"
                    return self._reply({"commitUrl": url, "commitOid": oid})
                self._reply({"error": "not found"}, 404)

            def do_GET(self):
                parts = self.path.split("?")[0].lstrip("/").split("/")
                if len(parts) >= 6 and parts[0] == "api" and parts[4] == "revision":
                    oid = hashlib.sha1(str(len(hub.commits)).encode()).hexdigest()
                    return self._reply({"id": f"{parts[2]}/{parts[3]}", "sha": oid, "private": False})
                self._reply({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
