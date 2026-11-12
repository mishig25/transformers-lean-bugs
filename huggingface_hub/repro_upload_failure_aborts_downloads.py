"""A failing `upload_folder` in one thread aborts an unrelated xet download running in another thread.

Real, unpatched huggingface_hub (main). Two actors in one process:
  - thread "download": hf_hub_download of a public xet file (distilbert-base-uncased model.safetensors, 268 MB) from
    the real Hub (anonymous) into a fresh temp cache.
  - thread "upload": HfApi(endpoint=<local fake Hub>).upload_folder(...) of a tiny folder. The fake Hub answers the
    preupload call with 403 (e.g. token without write access) -> _UploadPipeline._coordinator_loop raises ->
    _UploadPipeline.run's `except BaseException` calls abort_xet_session() (_upload_pipeline.py:454-457), which aborts
    the process-wide XetSession that the download thread is using.

Stubbed: only the Hub endpoint for the upload (a 30-line local HTTP server returning 403 on preupload).
Expected (correct) behaviour: the upload raises 403; the download is unaffected and completes.
"""

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from huggingface_hub import HfApi, hf_hub_download


REPO, FILE = "distilbert/distilbert-base-uncased", "model.safetensors"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


class FakeHub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = json.dumps({"error": "You don't have the rights to create a commit in this repo"}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Error-Message", "forbidden")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake_endpoint = f"http://127.0.0.1:{server.server_address[1]}"

    tmp = tempfile.mkdtemp(prefix="xetabort_")
    cache_dir = os.path.join(tmp, "cache")
    folder = os.path.join(tmp, "to_upload")
    os.makedirs(folder)
    with open(os.path.join(folder, "weights.bin"), "wb") as f:
        f.write(os.urandom(4096))

    result = {}

    def download():
        t0 = time.time()
        try:
            path = hf_hub_download(REPO, FILE, revision=REVISION, cache_dir=cache_dir)
            result["download"] = ("OK", path, time.time() - t0)
        except BaseException as e:  # noqa
            result["download"] = ("FAILED", f"{type(e).__name__}: {e}"[:300], time.time() - t0)

    def upload():
        try:
            HfApi(endpoint=fake_endpoint, token="hf_fake").upload_folder(
                repo_id="user/some-repo", folder_path=folder
            )
            result["upload"] = ("OK",)
        except BaseException as e:  # noqa
            result["upload"] = ("FAILED", f"{type(e).__name__}: {str(e).splitlines()[0]}"[:200])

    mode = sys.argv[1] if len(sys.argv) > 1 else "with_failing_upload"
    if os.environ.get("APPLY_FIX"):  # validate the proposed fix (fix_upload_pipeline_abort.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fix_upload_pipeline_abort

        fix_upload_pipeline_abort.apply()
    td = threading.Thread(target=download)
    td.start()
    # Wait until the xet download is transferring (temporary file has received data).
    blobs = os.path.join(cache_dir, "models--distilbert--distilbert-base-uncased", "blobs")
    deadline = time.time() + 120
    while time.time() < deadline and "download" not in result:
        try:
            sizes = [os.path.getsize(os.path.join(blobs, n)) for n in os.listdir(blobs) if n.endswith(".incomplete")]
        except OSError:
            sizes = []
        if any(sz > 1_000_000 for sz in sizes):
            break
        time.sleep(0.01)
    if mode == "with_failing_upload":
        tu = threading.Thread(target=upload)
        tu.start()
        tu.join()
        print("upload  :", result["upload"])
    td.join()
    print("download:", result["download"])
    if result["download"][0] == "OK":
        h = hashlib.sha256()
        with open(result["download"][1], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print("download sha256:", h.hexdigest())
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
