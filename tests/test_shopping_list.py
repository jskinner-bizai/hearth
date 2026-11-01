"""Shopping-list transaction and direct-handler tests; no sockets or providers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import sqlite3
import tempfile
import threading
import unittest

from hearth import Store, HearthError, DINNERS
from server import make_handler


class ShoppingBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hearth-shopping-test-")
        self.path = Path(self.temp.name) / "pantry.sqlite3"
        self.now = 1_800_000_000.0
        self.store = Store(self.path, clock=lambda: self.now)
        items = [{"name": "rice", "quantity": "100", "unit": "g"},
                 {"name": "spinach", "quantity": None, "unit": "g"},
                 {"name": "pasta", "quantity": "2", "unit": "each"}]
        self.store.confirm(self.store.preview_inventory(0, items)["id"])

    def tearDown(self):
        self.temp.cleanup()

    def preview(self, dinners=None):
        state = self.store.state()
        return self.store.preview_shopping_list(state["revision"], dinners if dinners is not None else
                                               ["tomato-pasta", "bean-rice-bowl"], state["shopping_revision"])

    def confirm(self, preview):
        return self.store.confirm_shopping_list(preview["id"], preview["content_hash"])

    def error(self, code, operation):
        with self.assertRaises(HearthError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)


class ShoppingTests(ShoppingBase):
    def test_preview_preserves_current_list_and_inventory(self):
        before = self.store.state()
        preview = self.preview()
        self.assertEqual(self.store.state(), before)
        rows = {r["name"]: r for r in preview["list"]["shopping"]}
        self.assertEqual((rows["rice"]["needed"], rows["rice"]["status"]), ("50", "add"))
        self.assertEqual(rows["spinach"]["status"], "check_quantity")
        self.assertIsNone(rows["spinach"]["needed"])
        self.assertEqual(rows["pasta"]["status"], "check_unit")
        self.assertEqual(rows["tomatoes"]["status"], "check_presence")

    def test_save_is_durable_separate_from_inventory(self):
        before = self.store.state()
        preview = self.preview()
        result = self.confirm(preview)
        after = Store(self.path).state()
        for key in ("revision", "items", "receipts"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["shopping_revision"], 1)
        self.assertEqual(after["shopping_list"], result["receipt"])
        self.assertEqual(result["receipt"]["list"], preview["list"])
        self.assertEqual(result["receipt"]["action"], "save_shopping_list")

    def test_empty_list_is_explicit_snapshot_not_inventory_clear(self):
        self.confirm(self.preview())
        before = self.store.state()
        empty = self.preview([])
        self.assertEqual(empty["list"]["selected_dinner_ids"], [])
        self.assertEqual(empty["list"]["shopping"], [])
        self.confirm(empty)
        state = self.store.state()
        self.assertEqual(state["shopping_revision"], 2)
        self.assertEqual(len(state["shopping_receipts"]), 2)
        self.assertEqual(state["items"], before["items"])

    def test_stale_pantry_and_list_previews_refused(self):
        preview = self.preview()
        self.store.confirm(self.store.preview_inventory(1, []) ["id"])
        before = self.store.state()
        self.error("stale_revision", lambda: self.confirm(preview))
        self.assertEqual(self.store.state(), before)
        self.error("stale_revision", lambda: self.store.preview_shopping_list(1, [], 0))
        first, second = self.preview([]), self.preview(["tomato-pasta"])
        self.confirm(first)
        self.error("stale_list_revision", lambda: self.confirm(second))
        self.error("stale_list_revision", lambda: self.store.preview_shopping_list(2, [], 0))

    def test_expiry_reverse_clock_and_payload_hash_fail_closed(self):
        preview = self.preview()
        self.error("preview_mismatch", lambda: self.store.confirm_shopping_list(preview["id"], "0" * 64))
        self.now -= 1
        self.error("clock_unavailable", lambda: self.confirm(preview))
        self.now += 601
        self.error("preview_expired", lambda: self.confirm(preview))
        self.assertEqual(self.store.state()["shopping_revision"], 0)

    def test_replay_returns_original_without_replacing_new_current(self):
        old = self.preview()
        receipt = self.confirm(old)["receipt"]
        self.confirm(self.preview(["tomato-pasta"]))
        self.store.confirm(self.store.preview_inventory(1, []) ["id"])
        self.now += 5000
        before = self.store.state()
        replay = self.confirm(old)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["receipt"], receipt)
        self.assertEqual(self.store.state(), before)
        self.error("preview_mismatch", lambda: self.store.confirm_shopping_list(old["id"], "f" * 64))

    def test_atomic_rollback_when_receipt_insert_fails(self):
        self.confirm(self.preview())
        preview = self.preview([])
        before = self.store.state()
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("CREATE TRIGGER reject_list BEFORE INSERT ON shopping_receipts BEGIN SELECT RAISE(ABORT,'injected'); END")
            con.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.confirm(preview)
        self.assertEqual(self.store.state(), before)

    def test_corrupted_snapshot_and_changed_recipe_refused(self):
        preview = self.preview()
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("UPDATE shopping_previews SET payload='{}' WHERE id=?", (preview["id"],))
            con.commit()
        self.error("preview_changed", lambda: self.confirm(preview))
        preview = self.preview()
        changed = json.loads(json.dumps(DINNERS)); changed[0]["ingredients"][0]["quantity"] = "201"
        with patch("hearth.DINNERS", changed):
            self.error("preview_changed", lambda: self.confirm(preview))
        self.assertEqual(self.store.state()["shopping_revision"], 0)

    def test_concurrent_distinct_list_saves_have_one_winner(self):
        one, two = self.preview([]), self.preview(["tomato-pasta"])
        barrier = threading.Barrier(2)
        def save(preview):
            barrier.wait()
            try:
                return Store(self.path, clock=lambda: self.now).confirm_shopping_list(preview["id"], preview["content_hash"])
            except HearthError as exc:
                return exc.code
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(save, [one, two]))
        self.assertEqual(sum(isinstance(v, dict) for v in results), 1)
        self.assertIn("stale_list_revision", results)
        self.assertEqual(self.store.state()["shopping_revision"], 1)

    def test_same_preview_concurrent_retry_has_one_receipt(self):
        preview = self.preview()
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.confirm(preview), range(2)))
        self.assertEqual(sorted(r["replayed"] for r in results), [False, True])
        self.assertEqual(results[0]["receipt"], results[1]["receipt"])
        self.assertEqual(self.store.state()["shopping_revision"], 1)

    def test_action_ids_and_invalid_fields_do_not_cross_boundaries(self):
        shopping = self.preview()
        inventory = self.store.preview_inventory(1, [])
        self.error("preview_missing", lambda: self.store.confirm(shopping["id"]))
        self.error("preview_missing", lambda: self.store.confirm_shopping_list(inventory["id"], shopping["content_hash"]))
        for bad in (None, True, -1, "1", 1.0):
            self.error("invalid_revision", lambda: self.store.preview_shopping_list(bad, [], 0))
            self.error("invalid_revision", lambda: self.store.preview_shopping_list(1, [], bad))
        for bad in (None, ["external"], ["tomato-pasta", "tomato-pasta"]):
            self.error("invalid_dinners", lambda: self.store.preview_shopping_list(1, bad, 0))


class DirectHandlerTests(ShoppingBase):
    """Exercise actual HTTP handler parsing/dispatch without binding any socket."""
    def request(self, route, data, *, extra=None, origin="http://127.0.0.1:8137"):
        body = json.dumps(data).encode()
        headers = {"Host": "127.0.0.1:8137", "Content-Type": "application/json", "Content-Length": str(len(body))}
        if origin is not None:
            headers["Origin"] = origin
        headers.update(extra or {})
        request = ("POST " + route + " HTTP/1.1\r\n" + "".join(k + ": " + v + "\r\n" for k, v in headers.items()) + "\r\n").encode() + body
        class Connection:
            def __init__(self): self.output = bytearray()
            def makefile(self, *_args, **_kwargs): return io.BytesIO(request)
            def sendall(self, value): self.output.extend(value)
        connection = Connection()
        make_handler(self.store)(connection, ("127.0.0.1", 12345), SimpleNamespace(server_address=("127.0.0.1", 8137)))
        head, payload = bytes(connection.output).split(b"\r\n\r\n", 1)
        return int(head.split()[1]), json.loads(payload)

    def test_routes_save_and_replay_exact_list(self):
        before = self.store.state()
        status, body = self.request("/api/shopping/preview", {"expected_revision": 1, "dinner_ids": ["tomato-pasta"], "expected_list_revision": 0})
        self.assertEqual(status, 200)
        preview = body["preview"]
        payload = {"preview_id": preview["id"], "content_hash": preview["content_hash"]}
        status, saved = self.request("/api/shopping/confirm", payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.request("/api/shopping/confirm", payload)[1]["receipt"], saved["receipt"])
        self.assertEqual(self.store.state()["items"], before["items"])

    def test_routes_refuse_external_or_broad_authority(self):
        payload = {"expected_revision": 1, "dinner_ids": [], "expected_list_revision": 0}
        self.assertEqual(self.request("/api/shopping/preview", payload, origin=None)[0], 403)
        self.assertEqual(self.request("/api/shopping/preview", payload, origin="https://example.com")[0], 403)
        self.assertEqual(self.request("/api/shopping/preview", {**payload, "purchase": True})[0], 400)
        self.assertEqual(self.request("/api/shopping/preview", {**payload, "dinner_ids": None})[0], 400)
        self.assertEqual(self.request("/api/shopping/confirm", {"preview_id": "x", "content_hash": "x", "buy": True})[0], 400)
        self.assertEqual(self.request("/api/purchase", payload)[0], 404)
        self.assertEqual(self.store.state()["shopping_revision"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
