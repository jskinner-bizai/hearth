"""Fridge scan, version 2: Spark finds each item and boxes every unit it can see; SAM 3.1 confirms each box.

Why: on a real fridge photo, SAM concept prompts ("yogurt cup", "mustard bottle") often returned nothing, while
Spark's own boxes passed to SAM as box prompts came back with 0.75 to 0.97 confidence. Meta's docs recommend this
pattern (let Spark decide what to segment, hand it to SAM).
"""
from concurrent.futures import ThreadPoolExecutor
import io
import json

from PIL import Image

import meta_client as meta
from hearth import HearthError, item_key

CONFIRM_BELOW = 0.6
MAX_ITEMS = 20
MAX_BOXES_PER_ITEM = 6
SAM_BATCH = 16

SCAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items", "scene_note"],
    "properties": {
        "scene_note": {"type": "string"},
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "unit", "quantity", "certainty", "boxes"],
            "properties": {
                "name": {"type": "string"},
                "unit": {"type": "string", "enum": ["g", "ml", "each"]},
                "quantity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "certainty": {"type": "string", "enum": ["clear", "partly_hidden", "guess"]},
                "boxes": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}}}}}

SCAN_SYSTEM = (
    "You inventory a household fridge or pantry from a photo. List each distinct food item you can see. "
    "name: what a parent would write on a shopping list ('whole milk', 'eggs', 'hot sauce', 'leftover pasta'). "
    "unit: 'each' for countable things, 'ml' for liquids, 'g' otherwise. "
    "quantity: a plain number in that unit only if it is printed and readable, or for 'each' the number of units you "
    "can see; otherwise null. Never guess weights or volumes. "
    "certainty: 'clear', 'partly_hidden' or 'guess'. "
    "boxes: one box per visible unit as [x1,y1,x2,y2] normalized 0 to 1000 (at most 6 per item). "
    "Skip non-food items, appliances, magnets and photos. At most 20 items.")


def _sam_boxes(raw, mime, w, h, boxes):
    """Send normalized boxes to SAM as box prompts, in batches of 16. Returns {object_id: match}."""
    def to_px(b):
        x1, y1, x2, y2 = [max(0, min(1000, int(v))) for v in b[:4]]
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1 * w // 1000, y1 * h // 1000, min(w - 1, x2 * w // 1000), min(h - 1, y2 * h // 1000))

    part = meta.sam_image_part(raw, mime)
    batches, current = [], []
    for oid, b in boxes:
        px = to_px(b)
        if px is None:
            continue
        current.append("%d<|box;x1=%d;y1=%d;x2=%d;y2=%d;w=%d;h=%d|>" % ((oid,) + px + (w, h)))
        if len(current) == SAM_BATCH:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    def run(tokens):
        body = {"model": meta.SAM, "stream": True, "metadata": {"include_confidence": "true"},
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "<0f>" + ",".join(tokens)}, part]}]}
        text = meta._collect_stream(meta._post("/responses", json.dumps(body).encode(), accept="text/event-stream"))
        return meta.parse_sam(text)

    found = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for matches in pool.map(run, batches):
            for m in matches:
                found[m["object"]] = m
    return found


def scan_image(raw, mime, clean_quantity):
    if mime not in ("image/jpeg", "image/png", "image/webp") or not 1000 < len(raw) <= 8_000_000:
        raise HearthError("invalid_image", "Send a JPEG, PNG or WebP photo under 8 MB.", 400)
    try:
        w, h = Image.open(io.BytesIO(raw)).size
    except Exception:
        raise HearthError("invalid_image", "That file could not be read as a photo.", 400)
    listed, usage = meta.spark_json(SCAN_SYSTEM, [
        {"type": "text", "text": "Inventory the food in this photo."}, meta.image_part(raw, mime)],
        "pantry_scan", SCAN_SCHEMA)
    items = listed.get("items", [])[:MAX_ITEMS]

    boxes, owner, next_id = [], {}, 1
    for idx, item in enumerate(items):
        for b in (item.get("boxes") or [])[:MAX_BOXES_PER_ITEM]:
            if isinstance(b, list) and len(b) >= 4:
                boxes.append((next_id, b))
                owner[next_id] = idx
                next_id += 1
    sam_error = None
    try:
        found = _sam_boxes(raw, mime, w, h, boxes) if boxes else {}
    except meta.MetaError as err:
        found, sam_error = {}, err.code

    draft, seen = [], set()
    for idx, item in enumerate(items):
        name = " ".join((item.get("name") or "").split())[:80]
        if not name or item_key(name) in seen:
            continue
        seen.add(item_key(name))
        mine = [found[oid] for oid, o in owner.items() if o == idx and oid in found]
        asked = sum(1 for o in owner.values() if o == idx)
        confirmed = [m for m in mine if (m["confidence"] or 0) >= CONFIRM_BELOW]
        unit = item.get("unit") if item.get("unit") in ("g", "ml", "each") else "each"
        quantity = clean_quantity(item.get("quantity"), unit)
        counted_by = None
        reasons = []
        if unit == "each" and confirmed:
            quantity, counted_by = str(len(confirmed)), "sam"
        if item.get("quantity") is not None and quantity is None:
            reasons.append("package size unclear")
        if asked == 0:
            reasons.append("no location given")
        elif not confirmed:
            reasons.append("SAM could not confirm it")
        elif len(confirmed) < asked:
            reasons.append("SAM confirmed %d of %d" % (len(confirmed), asked))
        if item.get("certainty") == "partly_hidden":
            reasons.append("partly hidden")
        elif item.get("certainty") == "guess":
            reasons.append("Spark was guessing")
        conf = max((m["confidence"] for m in mine if m["confidence"] is not None), default=None)
        draft.append({"name": name, "quantity": quantity, "unit": unit, "counted_by": counted_by,
                      "sam_matches": len(confirmed), "sam_confidence": conf,
                      "boxes": [m["box"] for m in mine], "image_size": {"w": w, "h": h},
                      "needs_check": bool(reasons), "reasons": reasons, "sam_error": sam_error})
    draft.sort(key=lambda d: (not d["needs_check"], item_key(d["name"])))
    return {"items": draft, "scene_note": listed.get("scene_note", ""), "models": [meta.SPARK, meta.SAM],
            "usage": usage, "qualification": "Draft only. Nothing is saved until you review and confirm the pantry."}
