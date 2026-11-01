"""Hearth HTTP server. Loopback by default; HEARTH_PUBLIC_HOST allows one public hostname behind HTTPS."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import os
import threading
import time
import json
import sqlite3
import sys
from hearth import Store, HearthError
import agents
import meta_client

PUBLIC_HOST = os.environ.get("HEARTH_PUBLIC_HOST", "").strip().lower()
AI_LIMIT_PER_MIN = int(os.environ.get("HEARTH_AI_PER_MIN", "12"))
AI_LIMIT_PER_DAY = int(os.environ.get("HEARTH_AI_PER_DAY", "400"))
_ai_calls, _ai_day, _ai_lock = [], [], threading.Lock()


def ai_budget():
    """Global per-minute and per-day limits so a public demo cannot run up the Meta bill."""
    now = time.time()
    with _ai_lock:
        _ai_calls[:] = [t for t in _ai_calls if now - t < 60]
        _ai_day[:] = [t for t in _ai_day if now - t < 86400]
        if len(_ai_day) >= AI_LIMIT_PER_DAY:
            raise HearthError("daily_limit", "Hearth's demo has hit its daily AI limit. Try again tomorrow.", 429)
        if len(_ai_calls) >= AI_LIMIT_PER_MIN:
            raise HearthError("busy", "Hearth is handling a lot of requests. Try again in a minute.", 429)
        _ai_calls.append(now)
        _ai_day.append(now)

ROOT = Path(__file__).resolve().parent


def reject_constant(_):
    raise ValueError("Non-finite JSON number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def _host(self):
            port = self.server.server_address[1]
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} and not (PUBLIC_HOST and host == PUBLIC_HOST):
                raise HearthError("host_rejected", "Use the Hearth address.", 403)
            return host

        def _origin_ok(self, host):
            scheme = "https" if PUBLIC_HOST and host == PUBLIC_HOST else "http"
            return self.headers.get("Origin") == f"{scheme}://{host}"

        def _send(self, status, data, content_type="application/json; charset=utf-8"):
            data = data if isinstance(data, bytes) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            for name, value in {"Content-Type": content_type, "Content-Length": str(len(data)),
                                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                "X-Frame-Options": "DENY",
                                "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, error):
            self._send(error.status, {"error": {"code": error.code, "message": error.message}})

        def do_GET(self):
            try:
                self._host()
                route = urlsplit(self.path).path
                if route == "/":
                    self._send(200, (ROOT / "static/index.html").read_bytes(), "text/html; charset=utf-8")
                elif route == "/api/state":
                    self._send(200, store.state())
                elif route == "/api/plan":
                    self._send(200, store.plan())
                else:
                    raise HearthError("not_found", "Page not found.", 404)
            except HearthError as error:
                self._error(error)
            except (sqlite3.Error, OSError):
                self._error(HearthError("local_read_failed", "Local data could not be read.", 503))

        def do_POST(self):
            try:
                host = self._host()
                if not self._origin_ok(host):
                    raise HearthError("origin_rejected", "Only the Hearth page can make changes.", 403)
                route = urlsplit(self.path).path
                ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if route in ("/api/ai/scan", "/api/ai/transcribe"):
                    try:
                        size = int(self.headers.get("Content-Length", "-1"))
                    except ValueError:
                        size = -1
                    if not 0 < size <= 40_000_000:
                        raise HearthError("body_size", "Send a photo or clip under 40 MB.", 413)
                    raw = self.rfile.read(size)
                    ai_budget()
                    if route == "/api/ai/scan":
                        if ctype.startswith("video/"):
                            import scan_video
                            self._send(200, scan_video.scan_video(raw, ctype))
                        else:
                            self._send(200, agents.scan_image(raw, ctype))
                    else:
                        if ctype not in ("audio/wav", "audio/x-wav", "audio/wave"):
                            raise HearthError("audio_format", "Send WAV audio.", 415)
                        self._send(200, agents.transcribe(store, raw))
                    return
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise HearthError("json_required", "Send JSON from the local page.", 415)
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if not 0 <= length <= 65536:
                    raise HearthError("body_size", "The request is missing or too large.", 413)
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8"),
                                      parse_constant=reject_constant, object_pairs_hook=unique_object)
                except (ValueError, UnicodeError):
                    raise HearthError("invalid_json", "The request is not valid JSON.")
                if not isinstance(data, dict):
                    raise HearthError("invalid_request", "The request must be an object.")
                route = urlsplit(self.path).path
                if route == "/api/preview":
                    if set(data) != {"expected_revision", "items"}:
                        raise HearthError("invalid_request", "Preview only the listed inventory data.")
                    self._send(200, {"preview": store.preview_inventory(data["expected_revision"], data["items"])})
                elif route == "/api/confirm":
                    if set(data) != {"preview_id"}:
                        raise HearthError("invalid_request", "Confirm one exact preview.")
                    self._send(200, store.confirm(data["preview_id"]))
                elif route == "/api/plan":
                    if set(data) != {"expected_revision", "dinner_ids"} or data["expected_revision"] is None or data["dinner_ids"] is None:
                        raise HearthError("invalid_request", "Choose local dinners against an exact saved pantry revision.")
                    self._send(200, store.plan(data["expected_revision"], data["dinner_ids"]))
                elif route == "/api/shopping/preview":
                    if set(data) != {"expected_revision", "dinner_ids", "expected_list_revision"} or data["dinner_ids"] is None:
                        raise HearthError("invalid_request", "Review one exact dinner selection and saved pantry/list revision.")
                    self._send(200, {"preview": store.preview_shopping_list(
                        data["expected_revision"], data["dinner_ids"], data["expected_list_revision"])})
                elif route == "/api/ai/plan":
                    if set(data) - {"request"} or not isinstance(data.get("request", ""), str):
                        raise HearthError("invalid_request", "Ask with plain text.")
                    ai_budget()
                    self._send(200, agents.plan_dinners(store, data.get("request", "")))
                elif route == "/api/shopping/confirm":
                    if set(data) != {"preview_id", "content_hash"}:
                        raise HearthError("invalid_request", "Save only the exact reviewed local shopping list.")
                    self._send(200, store.confirm_shopping_list(data["preview_id"], data["content_hash"]))
                else:
                    raise HearthError("not_found", "Action not found.", 404)
            except HearthError as error:
                self._error(error)
            except meta_client.MetaError as error:
                print(f"{time.strftime('%H:%M:%S')} meta error {error.status} {error.code}: {error.message}", file=sys.stderr, flush=True)
                self._send(error.status, {"error": {"code": error.code, "message": error.message}})
            except sqlite3.Error:
                if urlsplit(self.path).path == "/api/plan":
                    self._error(HearthError("local_read_failed", "The saved pantry could not be read. Refresh it before planning.", 503))
                else:
                    self._error(HearthError("local_save_failed", "Local save failed. Check state, then retry the same preview.", 503))
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
                self._error(HearthError("unexpected", "Something went wrong on Hearth's side. Try that again.", 500))

        def log_message(self, *_):
            pass
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--db", default=str(ROOT / "runtime/hearth.sqlite3"))
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(Store(args.db)))
    print(f"Hearth: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
