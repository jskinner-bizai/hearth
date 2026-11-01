"""Hearth state engine. Models propose; this module validates, previews and records confirmed changes."""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import hashlib
import math
import re
import sqlite3
import time
import unicodedata
import uuid


class HearthError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


PROVIDER = {"status": "meta_model_api", "models": ["muse-spark-1.3", "sam-3.1", "muse-voice-transcribe-1.0"],
            "message": "Meta models draft and propose. Hearth validates and you confirm."}
DINNERS = [
    {"id": "tomato-pasta", "name": "Tomato pasta", "ingredients": [
        {"name": "pasta", "quantity": "200", "unit": "g"},
        {"name": "tomatoes", "quantity": "400", "unit": "g"},
        {"name": "spinach", "quantity": "100", "unit": "g"}]},
    {"id": "bean-rice-bowl", "name": "Bean rice bowl", "ingredients": [
        {"name": "rice", "quantity": "150", "unit": "g"},
        {"name": "beans", "quantity": "240", "unit": "g"},
        {"name": "spinach", "quantity": "100", "unit": "g"}]}]


SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")


def validate_dinners(dinners):
    """Validate a proposed dinner set (from Muse Spark or the local examples)."""
    if not isinstance(dinners, list) or not 1 <= len(dinners) <= 3:
        raise HearthError("invalid_dinner_set", "A dinner proposal needs one to three dinners.")
    out, ids = [], set()
    for d in dinners:
        if not isinstance(d, dict) or not isinstance(d.get("id"), str) or not SLUG.fullmatch(d["id"]) or d["id"] in ids:
            raise HearthError("invalid_dinner_set", "Each proposed dinner needs a unique short id.")
        ids.add(d["id"])
        name = d.get("name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise HearthError("invalid_dinner_set", "Each proposed dinner needs a name.")
        ingredients = d.get("ingredients")
        if not isinstance(ingredients, list) or not 1 <= len(ingredients) <= 12:
            raise HearthError("invalid_dinner_set", "Each dinner needs one to twelve ingredients.")
        clean, seen = [], set()
        for ing in ingredients:
            if not isinstance(ing, dict) or ing.get("unit") not in ("g", "ml", "each"):
                raise HearthError("invalid_dinner_set", "Ingredient units must be g, ml or each.")
            iname = ing.get("name")
            if not isinstance(iname, str) or not 1 <= len(iname.strip()) <= 80 or item_key(iname.strip()) in seen:
                raise HearthError("invalid_dinner_set", "Ingredient names must be present and not repeated.")
            seen.add(item_key(iname.strip()))
            try:
                q = Decimal(str(ing.get("quantity")))
                if not q.is_finite() or not 0 < q <= 100000:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                raise HearthError("invalid_dinner_set", "Ingredient amounts must be positive numbers.")
            clean.append({"name": unicodedata.normalize("NFKC", iname.strip()), "quantity": format(q.normalize(), "f"),
                          "unit": ing["unit"]})
        entry = {"id": d["id"], "name": name.strip(), "ingredients": clean}
        if isinstance(d.get("why"), str) and d["why"].strip():
            entry["why"] = d["why"].strip()[:280]
        out.append(entry)
    return out


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def item_key(name):
    return unicodedata.normalize("NFKC", name).casefold()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_items(items):
    if not isinstance(items, list) or len(items) > 100:
        raise HearthError("invalid_items", "Provide at most 100 pantry items.")
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "quantity", "unit"}:
            raise HearthError("invalid_item", "Each item needs only name, quantity and unit.")
        name = item["name"]
        if not isinstance(name, str):
            raise HearthError("invalid_name", "An item name must be text.")
        name = unicodedata.normalize("NFKC", name.strip())
        if not 1 <= len(name) <= 80 or any(unicodedata.category(c).startswith("C") for c in name):
            raise HearthError("invalid_name", "Use a name of 1 to 80 visible characters.")
        key = item_key(name)
        if key in seen:
            raise HearthError("duplicate_item", "Correct duplicate ingredient names before previewing.")
        seen.add(key)
        if item["unit"] not in ("g", "ml", "each"):
            raise HearthError("invalid_unit", "Use g, ml or each; units are never guessed.")
        value = item["quantity"]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise HearthError("invalid_quantity", "Enter an amount or leave it unknown.")
            try:
                quantity = Decimal(str(value))
                if not quantity.is_finite() or not 0 <= quantity <= 1_000_000 or quantity.as_tuple().exponent < -3:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                raise HearthError("invalid_quantity", "Use a finite amount from 0 to 1000000, at most 3 decimal places.")
            value = format(quantity.normalize(), "f")
        result.append({"name": name, "quantity": value, "unit": item["unit"]})
    return sorted(result, key=lambda x: item_key(x["name"]))


class Store:
    def __init__(self, path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO meta VALUES (1,0);
                CREATE TABLE IF NOT EXISTS inventory (item_key TEXT PRIMARY KEY, name TEXT NOT NULL, quantity TEXT, unit TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS previews (id TEXT PRIMARY KEY, action TEXT NOT NULL, base_revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, preview_id TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL, from_revision INTEGER NOT NULL, to_revision INTEGER NOT NULL,
                    confirmed REAL NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS shopping_meta (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL);
                INSERT OR IGNORE INTO shopping_meta VALUES (1,0);
                CREATE TABLE IF NOT EXISTS shopping_previews (id TEXT PRIMARY KEY, inventory_revision INTEGER NOT NULL,
                    list_revision INTEGER NOT NULL, payload TEXT NOT NULL, content_hash TEXT NOT NULL,
                    created REAL NOT NULL, expires REAL NOT NULL, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS dinner_sets (id TEXT PRIMARY KEY, inventory_revision INTEGER NOT NULL,
                    source TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL, content_hash TEXT NOT NULL,
                    note TEXT);
                CREATE TABLE IF NOT EXISTS shopping_receipts (id TEXT PRIMARY KEY, preview_id TEXT NOT NULL UNIQUE,
                    from_revision INTEGER NOT NULL, to_revision INTEGER NOT NULL, inventory_revision INTEGER NOT NULL,
                    confirmed REAL NOT NULL, payload TEXT NOT NULL, content_hash TEXT NOT NULL);
            """)

    @contextmanager
    def _db(self, write=False):
        con = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        con.row_factory = sqlite3.Row
        try:
            if write:
                con.execute("BEGIN IMMEDIATE")
            yield con
            if write:
                con.commit()
        except Exception:
            if con.in_transaction:
                con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _items(con):
        return [dict(row) for row in con.execute("SELECT name,quantity,unit FROM inventory ORDER BY item_key")]

    @staticmethod
    def _receipt(row):
        return {"id": row["id"], "preview_id": row["preview_id"], "action": row["action"],
                "from_revision": row["from_revision"], "to_revision": row["to_revision"],
                "confirmed_at": stamp(row["confirmed"]), "items": json.loads(row["payload"])}

    @staticmethod
    def _shopping_receipt(row):
        return {"id": row["id"], "preview_id": row["preview_id"], "action": "save_shopping_list",
                "from_list_revision": row["from_revision"], "to_list_revision": row["to_revision"],
                "inventory_revision": row["inventory_revision"], "confirmed_at": stamp(row["confirmed"]),
                "content_hash": row["content_hash"], "list": json.loads(row["payload"])}

    @staticmethod
    def _dinner_set(con):
        row = con.execute("SELECT * FROM dinner_sets ORDER BY created DESC, rowid DESC LIMIT 1").fetchone()
        if row is None:
            return {"id": None, "source": "local_examples", "inventory_revision": None, "note": None,
                    "dinners": json.loads(json.dumps(DINNERS))}
        return {"id": row["id"], "source": row["source"], "inventory_revision": row["inventory_revision"],
                "note": row["note"], "dinners": json.loads(row["payload"])}

    def dinner_set(self):
        with self._db() as con:
            return self._dinner_set(con)

    def save_dinner_set(self, expected_revision, dinners, source, note=None):
        """Record a validated proposal bound to the pantry revision it was planned against.

        This never touches inventory or the shopping list; it only changes which dinners the preview uses.
        """
        if type(expected_revision) is not int or expected_revision < 0:
            raise HearthError("invalid_revision", "Refresh the current pantry revision.")
        dinners = validate_dinners(dinners)
        ident, now = str(uuid.uuid4()), self.clock()
        with self._db(write=True) as con:
            if con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0] != expected_revision:
                raise HearthError("stale_revision", "The pantry changed while dinners were being planned. Ask again.", 409)
            con.execute("INSERT INTO dinner_sets VALUES (?,?,?,?,?,?,?)",
                        (ident, expected_revision, source, now, json.dumps(dinners), content_hash(dinners),
                         (note or "")[:400]))
        return {"id": ident, "source": source, "inventory_revision": expected_revision, "dinners": dinners}

    def state(self):
        with self._db() as con:
            con.execute("BEGIN")
            shopping_receipts = [self._shopping_receipt(r) for r in con.execute(
                "SELECT * FROM shopping_receipts ORDER BY to_revision DESC LIMIT 10")]
            return {"revision": con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0],
                    "items": self._items(con),
                    "receipts": [self._receipt(r) for r in con.execute("SELECT * FROM receipts ORDER BY to_revision DESC LIMIT 10")],
                    "shopping_revision": con.execute("SELECT revision FROM shopping_meta WHERE id=1").fetchone()[0],
                    "shopping_list": shopping_receipts[0] if shopping_receipts else None,
                    "shopping_receipts": shopping_receipts,
                    "recipe_hash": content_hash(self._dinner_set(con)["dinners"]),
                    "dinner_source": self._dinner_set(con)["source"],
                    "provider": dict(PROVIDER)}

    @staticmethod
    def _shopping_payload(plan):
        return {"action": "save_shopping_list", "inventory_revision": plan["revision"],
                "selected_dinner_ids": plan["selected_dinner_ids"], "shopping": plan["shopping"],
                "recipe_hash": plan["recipe_hash"], "mode": plan["mode"],
                "qualification": "Saved shopping checklist. Unknown stock, quantities and units remain checks. "
                                 "This does not change pantry stock, place an order or authorize any external action."}

    def preview_shopping_list(self, expected_revision, dinner_ids, expected_list_revision):
        for value in (expected_revision, expected_list_revision):
            if type(value) is not int or value < 0:
                raise HearthError("invalid_revision", "Refresh the saved pantry and shopping-list revisions.")
        if dinner_ids is None:
            raise HearthError("invalid_dinners", "Choose an explicit dinner selection, including an empty list if desired.")
        plan = self.plan(expected_revision, dinner_ids)
        payload = self._shopping_payload(plan)
        digest = content_hash(payload)
        now, ident = self.clock(), str(uuid.uuid4())
        if not math.isfinite(now):
            raise HearthError("clock_unavailable", "The local clock is unavailable.", 503)
        with self._db(write=True) as con:
            revision = con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0]
            list_revision = con.execute("SELECT revision FROM shopping_meta WHERE id=1").fetchone()[0]
            if revision != expected_revision:
                raise HearthError("stale_revision", "The pantry changed. Review a fresh shopping list.", 409)
            if list_revision != expected_list_revision:
                raise HearthError("stale_list_revision", "The saved shopping list changed. Refresh it before reviewing a replacement.", 409)
            con.execute("INSERT INTO shopping_previews VALUES (?,?,?,?,?,?,?,?)",
                        (ident, revision, list_revision, json.dumps(payload), digest, now, now + 600, "pending"))
        return {"id": ident, "action": "save_shopping_list", "inventory_revision": revision,
                "expected_list_revision": list_revision, "content_hash": digest, "list": payload,
                "created_at": stamp(now), "expires_at": stamp(now + 600), "state": "pending"}

    def confirm_shopping_list(self, preview_id, expected_content_hash):
        if not isinstance(preview_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", preview_id):
            raise HearthError("invalid_preview", "Choose the exact shopping-list preview to save.")
        if not isinstance(expected_content_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_content_hash):
            raise HearthError("invalid_preview", "Confirm the exact reviewed shopping-list content.")
        with self._db(write=True) as con:
            prior = con.execute("SELECT * FROM shopping_receipts WHERE preview_id=?", (preview_id,)).fetchone()
            if prior is not None:
                if prior["content_hash"] != expected_content_hash:
                    raise HearthError("preview_mismatch", "The confirmation does not match that original shopping list.", 409)
                return {"receipt": self._shopping_receipt(prior), "replayed": True}
            preview = con.execute("SELECT * FROM shopping_previews WHERE id=?", (preview_id,)).fetchone()
            if preview is None:
                raise HearthError("preview_missing", "This shopping-list preview was not found.", 404)
            if preview["state"] != "pending":
                raise HearthError("unsupported_action", "Only a pending local shopping-list save is supported.")
            if preview["content_hash"] != expected_content_hash:
                raise HearthError("preview_mismatch", "The shopping-list content changed. Review it again.", 409)
            now = self.clock()
            if not math.isfinite(now) or now < preview["created"]:
                raise HearthError("clock_unavailable", "The local clock changed. Review a fresh shopping list.", 503)
            if now >= preview["expires"]:
                raise HearthError("preview_expired", "This shopping-list preview expired. Review it again.", 410)
            revision = con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0]
            list_revision = con.execute("SELECT revision FROM shopping_meta WHERE id=1").fetchone()[0]
            if revision != preview["inventory_revision"]:
                raise HearthError("stale_revision", "The pantry changed. Review a fresh shopping list.", 409)
            if list_revision != preview["list_revision"]:
                raise HearthError("stale_list_revision", "Another shopping list was saved. Refresh before replacing it.", 409)
            try:
                payload = json.loads(preview["payload"])
                if (content_hash(payload) != expected_content_hash or
                        payload != self._shopping_payload(self.plan(revision, payload.get("selected_dinner_ids")))):
                    raise ValueError("reviewed list changed")
            except (ValueError, TypeError, AttributeError):
                raise HearthError("preview_changed", "The reviewed shopping-list data changed. Make a fresh preview.", 409)
            receipt_id = str(uuid.uuid4())
            con.execute("INSERT INTO shopping_receipts VALUES (?,?,?,?,?,?,?,?)",
                        (receipt_id, preview_id, list_revision, list_revision + 1, revision, now,
                         json.dumps(payload), expected_content_hash))
            con.execute("UPDATE shopping_meta SET revision=revision+1 WHERE id=1")
            con.execute("UPDATE shopping_previews SET state='confirmed' WHERE id=?", (preview_id,))
            receipt = self._shopping_receipt(con.execute("SELECT * FROM shopping_receipts WHERE id=?", (receipt_id,)).fetchone())
        return {"receipt": receipt, "replayed": False}

    def preview_inventory(self, expected_revision, items):
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise HearthError("invalid_revision", "Refresh the current pantry revision.")
        items = validate_items(items)
        now, ident = self.clock(), str(uuid.uuid4())
        if not math.isfinite(now):
            raise HearthError("clock_unavailable", "The local clock is unavailable.", 503)
        with self._db(write=True) as con:
            current = con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0]
            if current != expected_revision:
                raise HearthError("stale_revision", "The pantry changed. Review a fresh preview.", 409)
            old = {item_key(i["name"]): i for i in self._items(con)}
            new = {item_key(i["name"]): i for i in items}
            diff = {"added": [new[k]["name"] for k in new.keys() - old.keys()],
                    "changed": [new[k]["name"] for k in new.keys() & old.keys() if old[k] != new[k]],
                    "removed": [old[k]["name"] for k in old.keys() - new.keys()]}
            diff = {k: sorted(v, key=item_key) for k, v in diff.items()}
            con.execute("INSERT INTO previews VALUES (?,?,?,?,?,?,?)",
                        (ident, "replace_inventory", current, json.dumps(items), now, now + 600, "pending"))
        return {"id": ident, "action": "replace_inventory", "expected_revision": current,
                "items": items, "created_at": stamp(now), "expires_at": stamp(now + 600),
                "state": "pending", "diff": diff}

    def confirm(self, preview_id):
        if not isinstance(preview_id, str) or not re.fullmatch(r"[0-9a-f-]{36}", preview_id):
            raise HearthError("invalid_preview", "Choose the exact preview to confirm.")
        with self._db(write=True) as con:
            prior = con.execute("SELECT * FROM receipts WHERE preview_id=?", (preview_id,)).fetchone()
            if prior is not None:
                return {"receipt": self._receipt(prior), "replayed": True}
            preview = con.execute("SELECT * FROM previews WHERE id=?", (preview_id,)).fetchone()
            if preview is None:
                raise HearthError("preview_missing", "This preview was not found.", 404)
            if preview["action"] != "replace_inventory" or preview["state"] != "pending":
                raise HearthError("unsupported_action", "Only the reviewed local inventory action is supported.")
            now = self.clock()
            if not math.isfinite(now) or now < preview["created"]:
                raise HearthError("clock_unavailable", "The local clock changed. Review a fresh preview.", 503)
            if now >= preview["expires"]:
                raise HearthError("preview_expired", "This preview expired. Make a fresh preview.", 410)
            revision = con.execute("SELECT revision FROM meta WHERE id=1").fetchone()[0]
            if revision != preview["base_revision"]:
                raise HearthError("stale_revision", "The pantry changed. Review a fresh preview.", 409)
            items = validate_items(json.loads(preview["payload"]))
            con.execute("DELETE FROM inventory")
            con.executemany("INSERT INTO inventory VALUES (?,?,?,?)",
                            [(item_key(i["name"]), i["name"], i["quantity"], i["unit"]) for i in items])
            con.execute("UPDATE meta SET revision=revision+1 WHERE id=1")
            receipt_id = str(uuid.uuid4())
            con.execute("INSERT INTO receipts VALUES (?,?,?,?,?,?,?)",
                        (receipt_id, preview_id, "replace_inventory", revision, revision + 1, now, json.dumps(items)))
            con.execute("UPDATE previews SET state='confirmed' WHERE id=?", (preview_id,))
            receipt = self._receipt(con.execute("SELECT * FROM receipts WHERE id=?", (receipt_id,)).fetchone())
        return {"receipt": receipt, "replayed": False}

    def plan(self, expected_revision=None, dinner_ids=None):
        if expected_revision is not None and (isinstance(expected_revision, bool) or
                not isinstance(expected_revision, int) or expected_revision < 0):
            raise HearthError("invalid_revision", "Refresh the current pantry revision.")
        current = self.dinner_set()
        base = current["dinners"]
        if dinner_ids is None:
            dinner_ids = [d["id"] for d in base]
        known_ids = {d["id"] for d in base}
        if (not isinstance(dinner_ids, list) or len(dinner_ids) > len(base) or
                any(not isinstance(ident, str) or ident not in known_ids for ident in dinner_ids) or
                len(set(dinner_ids)) != len(dinner_ids)):
            raise HearthError("invalid_dinners", "Choose each current dinner at most once.")
        state = self.state()
        if expected_revision is not None and expected_revision != state["revision"]:
            raise HearthError("stale_revision", "The saved pantry changed. Refresh it before planning.", 409)
        stock = {item_key(i["name"]): i for i in state["items"]}

        def shopping_for(dinners):
            required, sources = {}, {}
            for dinner in dinners:
                for ingredient in dinner["ingredients"]:
                    key = (ingredient["name"], ingredient["unit"])
                    required[key] = required.get(key, Decimal(0)) + Decimal(ingredient["quantity"])
                    sources.setdefault(key, []).append(dinner["name"])
            shopping = []
            for (name, unit), amount in sorted(required.items()):
                found = stock.get(item_key(name))
                on_hand = found["quantity"] if found else None
                if not found:
                    needed, status = format(amount, "f"), "check_presence"
                elif found["unit"] != unit:
                    needed, status = None, "check_unit"
                elif found["quantity"] is None:
                    needed, status = None, "check_quantity"
                else:
                    have = Decimal(found["quantity"])
                    needed = format(max(Decimal(0), amount - have), "f")
                    status = "add" if amount > have else "have_enough"
                shopping.append({"name": name, "unit": unit, "required": format(amount, "f"),
                                 "on_hand": on_hand, "on_hand_unit": found["unit"] if found else None,
                                 "needed": needed, "status": status, "source_dinners": sources[(name, unit)]})
            return shopping

        dinners = json.loads(json.dumps(base))
        for dinner in dinners:
            dinner["selected"] = dinner["id"] in dinner_ids
            dinner["shopping"] = shopping_for([dinner])
            statuses = {row["status"] for row in dinner["shopping"]}
            dinner["readiness"] = ("check_pantry" if any(s.startswith("check_") for s in statuses)
                                   else "shopping_needed" if "add" in statuses else "covered_amounts")
        selected = [dinner for dinner in dinners if dinner["selected"]]
        return {"revision": state["revision"],
                "mode": "meta_proposal" if current["source"] == "muse-spark" else "deterministic_local",
                "provider": dict(PROVIDER), "dinner_set_id": current["id"], "dinner_source": current["source"],
                "planned_against_revision": current["inventory_revision"],
                "recipe_hash": content_hash(base),
                "selected_dinner_ids": [d["id"] for d in selected], "dinners": dinners,
                "shopping": shopping_for(selected),
                "qualification": "Dinners are proposals. Hearth recomputes every amount from the saved pantry. "
                                 "Amounts cover selected dinners only; stock is counted once. Not recorded does not mean "
                                 "physically missing. Check unknown amounts and units. No stock changes or purchases occur."}
