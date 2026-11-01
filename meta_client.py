"""Small stdlib client for the Meta Model API (Muse Spark, SAM 3.1, Muse Voice Transcribe).

The API key is read from META_API_KEY on each call and is never logged or returned.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request
import uuid

BASE_URL = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
SPARK = os.environ.get("HEARTH_SPARK_MODEL", "muse-spark-1.3")
SAM = "sam-3.1"
ASR = "muse-voice-transcribe-1.0"


class MetaError(Exception):
    def __init__(self, code, message, status=502):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _key():
    key = os.environ.get("META_API_KEY", "").strip()
    if not key:
        raise MetaError("meta_not_configured", "Meta Model API key is not configured.", 503)
    return key


def _post(path, body, content_type="application/json", accept="application/json", timeout=90):
    req = urllib.request.Request(BASE_URL + path, data=body, method="POST", headers={
        "Authorization": "Bearer " + _key(), "Content-Type": content_type, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:300]
        raise MetaError("meta_http_%d" % err.code, "Meta API returned %d: %s" % (err.code, detail), 502)
    except (urllib.error.URLError, TimeoutError) as err:
        raise MetaError("meta_unreachable", "Meta API could not be reached (%s)." % type(err).__name__, 504)


def data_url(raw, mime):
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))


# ---------------------------------------------------------------- Muse Spark

def spark_json(system, user_content, schema_name, schema, temperature=0.2, max_tokens=4000):
    """Chat Completions with constrained JSON output. user_content is a string or content-part list."""
    body = {"model": SPARK, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema_name, "schema": schema, "strict": True}}}
    raw = _post("/chat/completions", json.dumps(body).encode())
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return json.loads(content), data.get("usage", {})
    except (KeyError, IndexError, ValueError, TypeError):
        raise MetaError("spark_bad_output", "Muse Spark returned output Hearth could not read.")


def image_part(raw, mime):
    return {"type": "image_url", "image_url": {"url": data_url(raw, mime)}}


# ---------------------------------------------------------------- SAM 3.1

_BOX = re.compile(r"(\d+)\s*<\|box;([^|]*)\|>")
_FRAME = re.compile(r"<(\d+)f>")


def _kv(text):
    out = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def parse_sam(text):
    """Return [{frame, object, box:{x1,y1,x2,y2}, size:{w,h}, confidence}] from SAM text output.

    Masks are ignored on purpose: Hearth only needs where each match is and how sure SAM is.
    """
    results = []
    for line in text.splitlines() or [text]:
        frames = list(_FRAME.finditer(line))
        for m in _BOX.finditer(line):
            frame = 0
            for f in frames:
                if f.start() < m.start():
                    frame = int(f.group(1))
            kv = _kv(m.group(2))
            if not {"x1", "y1", "x2", "y2"} <= set(kv):
                continue
            results.append({"frame": frame, "object": int(m.group(1)),
                            "box": {k: int(kv[k]) for k in ("x1", "y1", "x2", "y2")},
                            "size": {"w": int(kv.get("w", 0)), "h": int(kv.get("h", 0))},
                            "confidence": kv.get("c")})
    return results


def _collect_stream(raw):
    """Join response.output_text.delta events; also accept a non-streamed Responses object."""
    text = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:].strip())
            except ValueError:
                continue
            if event.get("type") == "response.output_text.delta":
                text.append(event.get("delta", ""))
    if text:
        return "".join(text)
    try:
        data = json.loads(raw)
        for item in data.get("output", []):
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    text.append(part.get("text", ""))
        return "".join(text)
    except ValueError:
        return raw


def sam_concept(media_part, concept, timeout=120):
    """One noun-phrase concept on an image or uploaded video. Returns parsed matches."""
    body = {"model": SAM, "stream": True,
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept}, media_part]}]}
    raw = _post("/responses", json.dumps(body).encode(), accept="text/event-stream", timeout=timeout)
    return parse_sam(_collect_stream(raw))


def sam_points(media_part, width, height, points, timeout=120):
    """points: [(object_id, x, y, positive)] in media pixels. Used for tap-to-correct."""
    tokens = "".join("%d%s<|point;x=%d;y=%d;w=%d;h=%d|>" % (oid, "" if pos else "-", x, y, width, height)
                     for oid, x, y, pos in points)
    body = {"model": SAM, "stream": True, "metadata": {"include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "<0f>" + tokens}, media_part]}]}
    raw = _post("/responses", json.dumps(body).encode(), accept="text/event-stream", timeout=timeout)
    return parse_sam(_collect_stream(raw))


def sam_image_part(raw, mime):
    return {"type": "input_image", "image_url": data_url(raw, mime)}


def upload_file(raw, filename, mime):
    boundary = uuid.uuid4().hex
    body = (("--%s\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n"
             "--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
             "Content-Type: %s\r\n\r\n") % (boundary, boundary, filename, mime)).encode() + raw + \
        ("\r\n--%s--\r\n" % boundary).encode()
    raw_resp = _post("/files", body, "multipart/form-data; boundary=" + boundary, timeout=180)
    try:
        return json.loads(raw_resp)["id"]
    except (ValueError, KeyError):
        raise MetaError("upload_failed", "Video upload did not return a file id.")


# ---------------------------------------------------------------- Voice Transcribe

def transcribe(wav_bytes, keywords=(), mode="PUSH_TO_TALK"):
    if not (wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"):
        raise MetaError("audio_format", "Send 16-bit mono WAV audio.", 400)
    settings = {"model": ASR, "mode": mode, "audioEncoding": "WAV", "languageBias": ["English"],
                "keywords": [k for k in keywords if isinstance(k, str) and 0 < len(k) <= 40][:50]}
    boundary = uuid.uuid4().hex
    body = (("--%s\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n"
             % boundary) + json.dumps(settings) + "\r\n").encode() + \
        ("--%s\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"ask.wav\"\r\n"
         "Content-Type: audio/wav\r\n\r\n" % boundary).encode() + wav_bytes + ("\r\n--%s--\r\n" % boundary).encode()
    raw = _post("/asr/transcribe", body, "multipart/form-data; boundary=" + boundary, timeout=60)
    try:
        data = json.loads(raw)
        return {"transcript": data.get("transcript", ""), "turns": data.get("turns", []),
                "duration_ms": data.get("audioDurationMs")}
    except ValueError:
        raise MetaError("asr_bad_output", "Voice transcription returned unreadable output.")
