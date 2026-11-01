"""Hearth's model layer. Meta models look, listen and propose; hearth.Store decides what gets saved.

Nothing in this module writes inventory or the shopping list. Outputs are drafts the person reviews.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import json
import re

import meta_client as meta
from hearth import HearthError, item_key

CONFIRM_BELOW = 0.6   # SAM confidence under this, or no SAM match, means "please confirm"

SCAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items", "scene_note"],
    "properties": {
        "scene_note": {"type": "string"},
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "concept", "unit", "quantity", "visible_count", "certainty"],
            "properties": {
                "name": {"type": "string"},
                "concept": {"type": "string"},
                "unit": {"type": "string", "enum": ["g", "ml", "each"]},
                "quantity": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "visible_count": {"type": "integer"},
                "certainty": {"type": "string", "enum": ["clear", "partly_hidden", "guess"]}}}}}}

SCAN_SYSTEM = (
    "You inventory a household fridge or pantry from a photo. List each distinct food item you can see. "
    "name: what a parent would call it on a shopping list (e.g. 'whole milk', 'eggs', 'baby spinach'). "
    "concept: a short concrete noun phrase a segmentation model can find (e.g. 'egg', 'milk carton', 'bell pepper'); "
    "no negation, no 'or', no positions. If two items are the same kind but differ in color or flavor, put that word "
    "in the concept ('red bell pepper', 'strawberry yogurt'). unit: 'each' for countable things, 'ml' for liquids, 'g' otherwise. "
    "quantity: a plain number in the chosen unit (e.g. '3780' for a gallon of milk in ml), only if printed on the "
    "package or obviously countable; otherwise null. Never guess weights. "
    "certainty: 'clear', 'partly_hidden' or 'guess'. Skip non-food items. At most 20 items.")

PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["dinners", "spoken_answer", "assumptions"],
    "properties": {
        "spoken_answer": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "dinners": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "name", "why", "ingredients"],
            "properties": {
                "id": {"type": "string"}, "name": {"type": "string"}, "why": {"type": "string"},
                "ingredients": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "quantity", "unit", "from_pantry"],
                    "properties": {"name": {"type": "string"}, "quantity": {"type": "string"},
                                   "unit": {"type": "string", "enum": ["g", "ml", "each"]},
                                   "from_pantry": {"type": "boolean"}}}}}}}}}

PLAN_SYSTEM = (
    "You plan weeknight dinners for a busy household with a toddler, using what is confirmed in their pantry. "
    "Propose exactly two dinners that take about 30 minutes. Prefer ingredients already in the pantry and reuse "
    "pantry names exactly as written. Each ingredient needs an amount for the whole household and a unit of g, ml or "
    "each. Mark from_pantry true only if the exact pantry item exists. ids are short lowercase slugs with hyphens. "
    "why: one plain sentence. spoken_answer: two or three short sentences a parent hears in the kitchen, naming the "
    "two dinners and what to pick up, if anything. Do not give nutrition, diet or health advice. assumptions: list "
    "anything you assumed, such as pantry staples you could not see.")


TO_UNIT = {"ml": 1, "l": 1000, "liter": 1000, "litre": 1000, "gal": 3785, "gallon": 3785, "fl oz": 29.57,
           "g": 1, "kg": 1000, "oz": 28.35, "lb": 453.6}


def clean_quantity(value, unit):
    """Return a plain decimal string Hearth accepts, or None. Never invents an amount."""
    if value is None:
        return None
    text = str(value).strip().lower()
    try:
        q = Decimal(text)
        return format(q.normalize(), "f") if q.is_finite() and 0 <= q <= 1_000_000 else None
    except InvalidOperation:
        pass
    for label, factor in sorted(TO_UNIT.items(), key=lambda kv: -len(kv[0])):
        m = re.search(r"(\d+(?:\.\d+)?)\s*%s\b" % re.escape(label), text)
        if m and ((unit == "ml" and label in ("ml", "l", "liter", "litre", "gal", "gallon", "fl oz")) or
                  (unit == "g" and label in ("g", "kg", "oz", "lb"))):
            return str(round(Decimal(m.group(1)) * Decimal(str(factor))))
    return None


def scan_image(raw, mime):
    """Spark finds and boxes each item; SAM 3.1 confirms each box. See scan_v2.py."""
    import scan_v2
    return scan_v2.scan_image(raw, mime, clean_quantity)


def plan_dinners(store, request_text=""):
    """Spark proposes two dinners against the confirmed pantry. Hearth validates and records the proposal."""
    state = store.state()
    pantry = [{"name": i["name"], "quantity": i["quantity"], "unit": i["unit"]} for i in state["items"]]
    if not pantry:
        raise HearthError("empty_pantry", "Confirm a pantry first so dinners come from what you actually have.", 409)
    ask = (request_text or "What can we make for dinner with what we have?").strip()[:500]
    proposal, usage = meta.spark_json(PLAN_SYSTEM, json.dumps({
        "request": ask, "confirmed_pantry": pantry,
        "note": "quantity null means the household has it but did not record how much."}),
        "dinner_plan", PLAN_SCHEMA, temperature=0.5)
    proposal = json.loads(json.dumps(proposal))
    pantry_keys = {item_key(i["name"]) for i in pantry}
    corrected = 0
    for dinner in proposal.get("dinners", []):
        for ing in dinner.get("ingredients", []):
            truly = item_key(ing.get("name", "")) in pantry_keys
            if ing.get("from_pantry") != truly:
                corrected += 1
            ing.pop("from_pantry", None)
    saved = store.save_dinner_set(state["revision"], proposal.get("dinners", [])[:2], "muse-spark",
                                  note=ask)
    plan = store.plan(state["revision"], [d["id"] for d in saved["dinners"]])
    return {"dinner_set": saved, "plan": plan, "spoken_answer": proposal.get("spoken_answer", "")[:600],
            "hearth_check": hearth_check(plan["shopping"]),
            "assumptions": proposal.get("assumptions", [])[:6], "pantry_flags_corrected": corrected,
            "model": meta.SPARK, "usage": usage}


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def hearth_check(shopping):
    """A sentence built from Hearth's own arithmetic, spoken after the model's answer."""
    buy = [r["name"] for r in shopping if r["status"] in ("add", "check_presence")]
    check = [r["name"] for r in shopping if r["status"] in ("check_quantity", "check_unit")]
    parts = []
    if buy:
        parts.append("Hearth's count says pick up " + _join(buy) + ".")
    if check:
        parts.append("Check how much " + _join(check) + " you have, the pantry doesn't record it.")
    return " ".join(parts) or "Hearth's count agrees: everything is in the saved pantry."


def transcribe(store, wav):
    keywords = [i["name"] for i in store.state()["items"]][:50]
    return meta.transcribe(wav, keywords=keywords, mode="DIARIZATION")
