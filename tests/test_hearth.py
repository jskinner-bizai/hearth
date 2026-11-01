from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from hearth import Store, HearthError
from server import make_handler


def item(name="rice", quantity="150", unit="g"):
    return {"name": name, "quantity": quantity, "unit": unit}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hearth-test-")
        self.base = Path(self.temp.name).resolve()
        assert self.base.is_relative_to(Path(tempfile.gettempdir()).resolve())
        self.path = self.base / "pantry.sqlite3"
        self.now = 1_800_000_000.0
        self.store = Store(self.path, clock=lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def preview(self, items=None):
        return self.store.preview_inventory(self.store.state()["revision"], items if items is not None else [item()])

    def assert_error(self, code, fn):
        with self.assertRaises(HearthError) as caught:
            fn()
        self.assertEqual(caught.exception.code, code)

    def test_preview_leaves_confirmed_state_unchanged(self):
        before = self.store.state()
        preview = self.preview([item(quantity=None)])
        self.assertEqual(self.store.state(), before)
        self.assertEqual(preview["diff"]["added"], ["rice"])
        self.assertIsNone(preview["items"][0]["quantity"])

    def test_confirmation_persists_after_reopen(self):
        result = self.store.confirm(self.preview()["id"])
        state = Store(self.path).state()
        self.assertEqual(state["revision"], 1)
        self.assertEqual(state["items"], [item()])
        self.assertEqual(state["receipts"][0], result["receipt"])
        self.assertEqual(state["provider"]["status"], "meta_model_api")

    def test_stale_confirmation_preserves_winning_inventory(self):
        old, winner = self.preview([item("beans")]), self.preview([item("pasta")])
        self.store.confirm(winner["id"])
        self.assert_error("stale_revision", lambda: self.store.confirm(old["id"]))
        self.assertEqual(self.store.state()["items"], [item("pasta")])

    def test_stale_preview_is_refused(self):
        self.store.confirm(self.preview()["id"])
        self.assert_error("stale_revision", lambda: self.store.preview_inventory(0, [item()]))

    def test_expiry_boundary_and_clock_reversal_refuse_write(self):
        preview = self.preview()
        self.now -= 1
        self.assert_error("clock_unavailable", lambda: self.store.confirm(preview["id"]))
        self.now += 601
        self.assert_error("preview_expired", lambda: self.store.confirm(preview["id"]))
        self.assertEqual(self.store.state()["revision"], 0)

    def test_successful_retry_returns_original_receipt_after_later_edits(self):
        preview = self.preview()
        first = self.store.confirm(preview["id"])
        self.store.confirm(self.preview([item("beans")])["id"])
        self.now += 10000
        retry = self.store.confirm(preview["id"])
        self.assertEqual(retry["receipt"], first["receipt"])
        self.assertTrue(retry["replayed"])
        self.assertEqual(self.store.state()["revision"], 2)

    def test_unicode_duplicates_and_invalid_values_rejected(self):
        self.assert_error("duplicate_item", lambda: self.preview([item("rice"), item("ＲＩＣＥ")]))
        for bad in ["NaN", "Infinity", "-1", "0.0001", True, {}, ""]:
            with self.subTest(value=bad):
                self.assert_error("invalid_quantity", lambda: self.preview([item(quantity=bad)]))
        self.assert_error("invalid_unit", lambda: self.preview([item(unit="kg")]))
        self.assert_error("invalid_name", lambda: self.preview([item(name="rice\nexecute")]))
        self.assert_error("invalid_revision", lambda: self.store.preview_inventory(True, []))

    def test_transaction_rolls_back_if_receipt_write_fails(self):
        self.store.confirm(self.preview([item("rice")])["id"])
        before = self.store.state()
        candidate = self.preview([item("beans")])
        con = sqlite3.connect(self.path)
        try:
            con.execute("CREATE TRIGGER refuse_receipt BEFORE INSERT ON receipts BEGIN SELECT RAISE(ABORT,'injected failure'); END")
            con.commit()
        finally:
            con.close()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.confirm(candidate["id"])
        self.assertEqual(self.store.state(), before)

    def test_concurrent_competing_previews_have_one_winner(self):
        ids = [self.preview([item("rice")])["id"], self.preview([item("beans")])["id"]]
        barrier = threading.Barrier(2)
        def run(ident):
            barrier.wait()
            try:
                return self.store.confirm(ident)
            except HearthError as error:
                return error.code
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, ids))
        self.assertEqual(sum(isinstance(r, dict) for r in results), 1)
        self.assertIn("stale_revision", results)
        self.assertEqual(len(self.store.state()["receipts"]), 1)

    def test_concurrent_retry_has_exactly_one_receipt(self):
        ident = self.preview()["id"]
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(self.store.confirm, [ident, ident]))
        self.assertEqual(results[0]["receipt"]["id"], results[1]["receipt"]["id"])
        self.assertEqual(sorted(r["replayed"] for r in results), [False, True])
        self.assertEqual(self.store.state()["revision"], 1)

    def test_unknown_amount_or_mismatched_unit_is_not_guessed(self):
        self.store.confirm(self.preview([item("spinach", None), item("pasta", "2", "each")])["id"])
        rows = {r["name"]: r for r in self.store.plan()["shopping"]}
        self.assertEqual(rows["spinach"]["status"], "check_quantity")
        self.assertIsNone(rows["spinach"]["needed"])
        self.assertEqual(rows["pasta"]["status"], "check_unit")
        self.assertIsNone(rows["pasta"]["needed"])
        self.assertEqual(rows["beans"]["needed"], "240")

    def test_plan_adds_requirements_without_changing_pantry(self):
        self.store.confirm(self.preview([item("spinach", "150")])["id"])
        before = self.store.state()
        plan = self.store.plan()
        spinach = next(r for r in plan["shopping"] if r["name"] == "spinach")
        self.assertEqual((spinach["required"], spinach["needed"]), ("200", "50"))
        plan["dinners"].clear()
        self.assertEqual(len(self.store.plan()["dinners"]), 2)
        self.assertEqual(self.store.state(), before)

    def test_dinner_choice_changes_only_read_only_requirements(self):
        self.store.confirm(self.preview([item("spinach", "150"), item("rice", "100")])["id"])
        before = self.store.state()
        pasta = self.store.plan(1, ["tomato-pasta"])
        rows = {r["name"]: r for r in pasta["shopping"]}
        self.assertNotIn("rice", rows)
        self.assertEqual(rows["spinach"]["required"], "100")
        self.assertEqual(rows["spinach"]["needed"], "0")
        self.assertEqual(rows["spinach"]["source_dinners"], ["Tomato pasta"])
        self.assertEqual(pasta["selected_dinner_ids"], ["tomato-pasta"])
        both = self.store.plan(1, ["bean-rice-bowl", "tomato-pasta"])
        spinach = next(r for r in both["shopping"] if r["name"] == "spinach")
        self.assertEqual((spinach["required"], spinach["needed"]), ("200", "50"))
        self.assertEqual(spinach["source_dinners"], ["Tomato pasta", "Bean rice bowl"])
        self.assertEqual(self.store.state(), before)

    def test_plan_empty_selection_and_missing_stock_are_explicit(self):
        plan = self.store.plan(0, [])
        self.assertEqual(plan["shopping"], [])
        self.assertEqual(plan["selected_dinner_ids"], [])
        self.assertTrue(all(not dinner["selected"] for dinner in plan["dinners"]))
        self.assertTrue(all(dinner["readiness"] == "check_pantry" for dinner in plan["dinners"]))
        row = plan["dinners"][0]["shopping"][0]
        self.assertEqual(row["status"], "check_presence")
        self.assertIsNone(row["on_hand"])
        self.assertIsNone(row["on_hand_unit"])
        self.assertEqual(row["needed"], row["required"])
        self.assertEqual(self.store.state()["revision"], 0)

    def test_plan_preserves_unit_evidence_and_dinner_readiness(self):
        self.store.confirm(self.preview([item("pasta", "2", "each"), item("tomatoes", "400"),
                                        item("spinach", "200"), item("rice", "100"), item("beans", "240")])["id"])
        plan = self.store.plan(1, ["tomato-pasta", "bean-rice-bowl"])
        pasta, bowl = plan["dinners"]
        self.assertEqual(pasta["readiness"], "check_pantry")
        self.assertEqual(bowl["readiness"], "shopping_needed")
        row = next(r for r in plan["shopping"] if r["name"] == "pasta")
        self.assertEqual((row["on_hand"], row["on_hand_unit"], row["needed"]), ("2", "each", None))
        self.store.confirm(self.preview([item("pasta", "200"), item("tomatoes", "400"), item("spinach", "200")])["id"])
        self.assertEqual(self.store.plan(2, ["tomato-pasta"])["dinners"][0]["readiness"], "covered_amounts")

    def test_stale_and_invalid_plan_requests_never_write(self):
        self.store.confirm(self.preview()["id"])
        before = self.store.state()
        self.assert_error("stale_revision", lambda: self.store.plan(0, ["tomato-pasta"]))
        for bad in [True, -1, "1", 1.0]:
            self.assert_error("invalid_revision", lambda: self.store.plan(bad, []))
        for bad in ["tomato-pasta", ["tomato-pasta", "tomato-pasta"], ["external"], [{}], [True]]:
            self.assert_error("invalid_dinners", lambda: self.store.plan(1, bad))
        self.assertEqual(self.store.state(), before)

    def test_individual_dinner_rows_do_not_alias_combined_or_global_data(self):
        plan = self.store.plan()
        plan["dinners"][0]["shopping"][0]["source_dinners"].clear()
        plan["dinners"][0]["ingredients"][0]["quantity"] = "90000"
        pasta = next(row for row in plan["shopping"] if row["name"] == "pasta")
        self.assertEqual(pasta["source_dinners"], ["Tomato pasta"])
        self.assertEqual(self.store.plan()["dinners"][0]["ingredients"][0]["quantity"], "200")


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hearth-http-test-")
        assert Path(self.temp.name).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        self.store = Store(Path(self.temp.name) / "pantry.sqlite3")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, route, data=None, headers=None, raw=None):
        defaults = {"Origin": f"http://127.0.0.1:{self.port}", "Content-Type": "application/json"}
        defaults.update(headers or {})
        defaults = {k: v for k, v in defaults.items() if v is not None}
        conn = HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request(method, route, raw if raw is not None else json.dumps(data).encode() if data is not None else None, defaults)
        response = conn.getresponse()
        status, body = response.status, json.loads(response.read())
        conn.close()
        return status, body

    def test_api_workflow_and_receipt(self):
        status, state = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        status, body = self.request("POST", "/api/preview", {"expected_revision": state["revision"], "items": [item()]})
        self.assertEqual(status, 200)
        status, saved = self.request("POST", "/api/confirm", {"preview_id": body["preview"]["id"]})
        self.assertEqual(status, 200)
        self.assertEqual(saved["receipt"]["to_revision"], 1)
        self.assertEqual(self.request("GET", "/api/plan")[1]["mode"], "deterministic_local")

    def test_host_and_origin_variants_denied(self):
        for host in [f"127.0.0.1.evil:{self.port}", f"127.0.0.1.:{self.port}", f"localhost:{self.port}."]:
            self.assertEqual(self.request("GET", "/api/state", headers={"Host": host})[0], 403)
        for origin in [None, "https://example.com", "null"]:
            self.assertEqual(self.request("POST", "/api/preview", {}, {"Origin": origin})[0], 403)
        self.assertEqual(self.store.state()["revision"], 0)

    def test_plan_api_is_scoped_read_only_and_checks_fresh_revision(self):
        before = self.store.state()
        status, plan = self.request("POST", "/api/plan", {"expected_revision": 0, "dinner_ids": ["bean-rice-bowl"]})
        self.assertEqual(status, 200)
        self.assertEqual(plan["selected_dinner_ids"], ["bean-rice-bowl"])
        self.assertNotIn("pasta", [r["name"] for r in plan["shopping"]])
        self.assertEqual(self.store.state(), before)
        self.store.confirm(self.store.preview_inventory(0, [item()])["id"])
        self.assertEqual(self.request("POST", "/api/plan", {"expected_revision": 0, "dinner_ids": []})[0], 409)
        for bad in [{"expected_revision": None, "dinner_ids": []}, {"expected_revision": 1, "dinner_ids": None},
                    {"expected_revision": 1}, {"expected_revision": 1, "dinner_ids": [], "buy": True}]:
            self.assertEqual(self.request("POST", "/api/plan", bad)[0], 400)
        self.assertEqual(self.request("POST", "/api/plan", {"expected_revision": 1, "dinner_ids": []}, {"Origin": "https://evil.test"})[0], 403)

    def test_plan_database_failure_reports_read_failure_without_save_advice(self):
        before = self.store.state()
        with patch.object(self.store, "plan", side_effect=sqlite3.OperationalError("injected read failure")):
            status, body = self.request("POST", "/api/plan", {"expected_revision": 0, "dinner_ids": []})
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "local_read_failed")
        self.assertEqual(self.store.state(), before)

    def test_json_boundary_and_unexpected_actions(self):
        for raw in [b'{"expected_revision":0,"items":[],"items":[]}', b'{"expected_revision":NaN,"items":[]}']:
            self.assertEqual(self.request("POST", "/api/preview", raw=raw)[0], 400)
        self.assertEqual(self.request("POST", "/api/preview", {"expected_revision": 0, "items": [], "action": "buy"})[0], 400)
        self.assertEqual(self.request("POST", "/api/preview", {}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/execute", {})[0], 404)
        self.assertEqual(self.request("POST", "/api/preview", raw=b"x" * 65537)[0], 413)
        self.assertEqual(self.store.state()["revision"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
