"""Pan mode: the parent pans the phone across the fridge instead of taking one photo.

Muse Spark watches the clip and lists the foods. SAM 3.1 tracks each one through the video and gives every
physical object a stable ID, so a yogurt cup that shows up in 60 frames is still counted once.
Verified Nov 1 on a real 15 second pan: Spark listed 10 foods, SAM tracked 5 yogurt cups across 63 frames.
"""
from concurrent.futures import ThreadPoolExecutor

import meta_client as meta
from hearth import HearthError, item_key

MIN_FRAMES = 3          # an ID seen in fewer frames is treated as tracking noise
MAX_CONCEPTS = 8        # cost and latency cap per scan (about $0.08 of SAM video per concept on a 15 s clip)
MAX_BYTES = 40_000_000
VIDEO_TYPES = {"video/mp4": "mp4", "video/quicktime": "mov"}

VIDEO_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items", "scene_note"],
    "properties": {
        "scene_note": {"type": "string"},
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "concept", "unit", "count", "certainty"],
            "properties": {
                "name": {"type": "string"},
                "concept": {"type": "string"},
                "unit": {"type": "string", "enum": ["g", "ml", "each"]},
                "count": {"type": "integer"},
                "certainty": {"type": "string", "enum": ["clear", "partly_hidden", "guess"]}}}}}}

VIDEO_SYSTEM = (
    "A parent panned a phone across their fridge. List each distinct food you saw at any point. "
    "name: what they would write on a shopping list. concept: a short concrete noun phrase a segmentation model can "
    "track, such as 'yogurt cup' or 'soda can' (no negation, no 'or', no positions). unit: 'each' for countable "
    "things, 'ml' for liquids, 'g' otherwise. count: how many separate units you saw over the whole video, counting "
    "each physical object once even if it appears many times. certainty: 'clear', 'partly_hidden' or 'guess'. "
    "Skip non-food items. At most 15 items.")


def scan_video(raw, mime):
    if mime not in VIDEO_TYPES or not 10_000 < len(raw) <= MAX_BYTES:
        raise HearthError("invalid_video", "Send an MP4 or MOV clip under 40 MB.", 400)
    listed, usage = meta.spark_json(VIDEO_SYSTEM, [
        {"type": "text", "text": "What food is in this fridge?"},
        {"type": "video_url", "video_url": {"url": meta.data_url(raw, "video/mp4" if mime == "video/mp4" else mime)}}],
        "pan_scan", VIDEO_SCHEMA)
    items = listed.get("items", [])[:15]
    file_id = meta.upload_file(raw, "pan." + VIDEO_TYPES[mime], mime)
    part = {"type": "input_video", "file_id": file_id}

    # Track countable items first; that is where stable IDs matter.
    ranked = sorted(items, key=lambda i: (i.get("unit") != "each", -int(i.get("count") or 0)))
    concepts = []
    for i in ranked:
        c = (i.get("concept") or "").strip().lower()
        if c and c not in concepts:
            concepts.append(c)
    concepts = concepts[:MAX_CONCEPTS]

    def track(concept):
        try:
            return concept, meta.sam_concept(part, concept, timeout=240), None
        except meta.MetaError as err:
            return concept, [], err.code

    with ThreadPoolExecutor(max_workers=4) as pool:
        tracked = {c: (m, e) for c, m, e in pool.map(track, concepts)}

    draft, seen = [], set()
    for item in items:
        name = " ".join((item.get("name") or "").split())[:80]
        if not name or item_key(name) in seen:
            continue
        seen.add(item_key(name))
        concept = (item.get("concept") or "").strip().lower()
        unit = item.get("unit") if item.get("unit") in ("g", "ml", "each") else "each"
        spark_count = int(item.get("count") or 0)
        reasons, quantity, counted_by = [], None, None
        frames_by_id = {}
        matches, err = tracked.get(concept, ([], None))
        for m in matches:
            frames_by_id.setdefault(m["object"], set()).add(m["frame"])
        stable = sorted(oid for oid, fr in frames_by_id.items() if len(fr) >= MIN_FRAMES)
        conf = max((m["confidence"] for m in matches if m["confidence"] is not None), default=None)
        if concept not in tracked:
            reasons.append("not tracked (scan limit)")
        elif err:
            reasons.append("tracking failed")
        elif not stable:
            reasons.append("SAM could not track it")
        if unit == "each":
            if stable:
                quantity, counted_by = str(len(stable)), "sam_video"
                if spark_count and spark_count != len(stable):
                    reasons.append("Spark saw %d, SAM tracked %d" % (spark_count, len(stable)))
            elif spark_count:
                quantity, counted_by = str(spark_count), "spark"
        if item.get("certainty") == "partly_hidden":
            reasons.append("partly hidden")
        elif item.get("certainty") == "guess":
            reasons.append("Spark was guessing")
        draft.append({"name": name, "quantity": quantity, "unit": unit, "counted_by": counted_by,
                      "sam_matches": len(stable), "sam_confidence": conf,
                      "frames_tracked": len(set().union(*frames_by_id.values())) if frames_by_id else 0,
                      "boxes": [], "image_size": None, "needs_check": bool(reasons), "reasons": reasons,
                      "sam_error": err})
    draft.sort(key=lambda d: (not d["needs_check"], item_key(d["name"])))
    return {"items": draft, "scene_note": listed.get("scene_note", ""), "mode": "video",
            "models": [meta.SPARK, meta.SAM], "usage": usage,
            "qualification": "Draft only. Each tracked object is counted once across the video. Review, then confirm."}
