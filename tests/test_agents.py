"""Tests for the Meta model layer. Network calls are replaced with recorded-shape fakes."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agents
import meta_client
from hearth import HearthError, Store

PANTRY = [{"name": "eggs", "quantity": "6", "unit": "each"}, {"name": "rice", "quantity": "500", "unit": "g"},
          {"name": "spinach", "quantity": None, "unit": "g"}]
PROPOSAL = {"spoken_answer": "Egg fried rice or spinach rice bowls. Grab soy sauce.", "assumptions": ["You have oil."],
            "dinners": [
                {"id": "egg-fried-rice", "name": "Egg fried rice", "why": "Uses the eggs and rice you have.",
                 "ingredients": [{"name": "eggs", "quantity": "4", "unit": "each", "from_pantry": True},
                                 {"name": "rice", "quantity": "300", "unit": "g", "from_pantry": True},
                                 {"name": "soy sauce", "quantity": "30", "unit": "ml", "from_pantry": True}]},
                {"id": "spinach-rice-bowl", "name": "Spinach rice bowl", "why": "Quick and green.",
                 "ingredients": [{"name": "rice", "quantity": "300", "unit": "g", "from_pantry": True},
                                 {"name": "spinach", "quantity": "150", "unit": "g", "from_pantry": True}]}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "h.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def confirm(self, items):
        rev = self.store.state()["revision"]
        self.store.confirm(self.store.preview_inventory(rev, items)["id"])


class SamParserTests(unittest.TestCase):
    def test_parses_boxes_confidence_and_frames(self):
        text = ("<0f>1<|box;x1=1;y1=2;x2=30;y2=40;w=640;h=480;c=0.9|><|mask;x=0;y=0;data=!a|>\n"
                "<5f>1<|box;x1=3;y1=2;x2=33;y2=40;w=640;h=480;c=0.8|><|mask;x=0;y=0;data=!b|>,"
                "2<|box;x1=100;y1=2;x2=130;y2=40;w=640;h=480|><|mask;x=0;y=0;data=!c|>")
        out = meta_client.parse_sam(text)
        self.assertEqual([(m["frame"], m["object"]) for m in out], [(0, 1), (5, 1), (5, 2)])
        self.assertEqual(out[0]["confidence"], 0.9)
        self.assertIsNone(out[2]["confidence"])

    def test_stream_collection(self):
        raw = ('data: {"type":"response.output_text.delta","delta":"1<|box;x1=1;y1=1;"}\n'
               'data: {"type":"response.output_text.delta","delta":"x2=9;y2=9;w=10;h=10;c=0.7|>"}\n'
               'data: {"type":"response.completed"}\n')
        self.assertEqual(len(meta_client.parse_sam(meta_client._collect_stream(raw))), 1)

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict("os.environ", {"META_API_KEY": ""}):
            with self.assertRaises(meta_client.MetaError) as ctx:
                meta_client._key()
        self.assertEqual(ctx.exception.code, "meta_not_configured")


class ScanTests(unittest.TestCase):
    def _png(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (200, 100), "white").save(buf, "PNG")
        return buf.getvalue() + b"0" * 1200

    def test_spark_boxes_confirmed_by_sam(self):
        listed = {"scene_note": "fridge", "items": [
            {"name": "eggs", "unit": "each", "quantity": "6", "certainty": "clear",
             "boxes": [[0, 0, 100, 100], [100, 0, 200, 100], [200, 0, 300, 100]]},
            {"name": "whole milk", "unit": "ml", "quantity": "1 gal", "certainty": "clear", "boxes": [[400, 0, 600, 900]]},
            {"name": "yogurt", "unit": "each", "quantity": None, "certainty": "partly_hidden", "boxes": [[700, 0, 800, 100]]},
            {"name": "salsa", "unit": "ml", "quantity": None, "certainty": "clear", "boxes": []}]}
        hit = lambda oid, c: {"frame": 0, "object": oid, "box": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
                              "size": {"w": 200, "h": 100}, "confidence": c}
        seen_tokens = []

        def fake_boxes(raw, mime, w, h, boxes):
            seen_tokens.append((w, h, boxes))
            return {1: hit(1, 0.9), 2: hit(2, 0.95), 3: hit(3, 0.4), 4: hit(4, 0.97)}

        import scan_v2
        with mock.patch.object(meta_client, "spark_json", return_value=(listed, {})), \
                mock.patch.object(scan_v2, "_sam_boxes", side_effect=fake_boxes):
            out = agents.scan_image(self._png(), "image/png")
        self.assertEqual(seen_tokens[0][:2], (200, 100))
        by = {i["name"]: i for i in out["items"]}
        self.assertEqual(by["eggs"]["quantity"], "2")  # SAM confirmed 2 of 3 boxes
        self.assertIn("SAM confirmed 2 of 3", by["eggs"]["reasons"])
        self.assertEqual(by["whole milk"]["quantity"], "3785")
        self.assertFalse(by["whole milk"]["needs_check"])
        self.assertIn("SAM could not confirm it", by["yogurt"]["reasons"])
        self.assertIn("no location given", by["salsa"]["reasons"])
        self.assertTrue(out["items"][0]["needs_check"])

    def test_sam_failure_keeps_draft_but_flags_everything(self):
        listed = {"scene_note": "", "items": [
            {"name": "eggs", "unit": "each", "quantity": "6", "certainty": "clear", "boxes": [[0, 0, 100, 100]]}]}
        import scan_v2
        with mock.patch.object(meta_client, "spark_json", return_value=(listed, {})), \
                mock.patch.object(scan_v2, "_sam_boxes", side_effect=meta_client.MetaError("meta_http_500", "x")):
            out = agents.scan_image(self._png(), "image/png")
        self.assertEqual(out["items"][0]["sam_error"], "meta_http_500")
        self.assertTrue(out["items"][0]["needs_check"])
        self.assertEqual(out["items"][0]["quantity"], "6")

    def test_rejects_non_images(self):
        with self.assertRaises(HearthError):
            agents.scan_image(b"x" * 5000, "text/plain")


class PlanTests(Base):
    def test_spark_proposal_is_validated_and_recomputed(self):
        self.confirm(PANTRY)
        with mock.patch.object(meta_client, "spark_json", return_value=(PROPOSAL, {})):
            out = agents.plan_dinners(self.store, "dinner?")
        plan = out["plan"]
        self.assertEqual(plan["mode"], "meta_proposal")
        self.assertEqual(out["pantry_flags_corrected"], 1)  # soy sauce is not in the pantry
        rows = {r["name"]: r for r in plan["shopping"]}
        self.assertEqual(rows["rice"]["status"], "add")  # 600 g needed, 500 g recorded
        self.assertEqual(rows["rice"]["needed"], "100")
        self.assertEqual(rows["spinach"]["status"], "check_quantity")
        self.assertEqual(rows["soy sauce"]["status"], "check_presence")

    def test_empty_pantry_refused(self):
        with self.assertRaises(HearthError) as ctx:
            agents.plan_dinners(self.store, "")
        self.assertEqual(ctx.exception.code, "empty_pantry")

    def test_stale_proposal_refused(self):
        self.confirm(PANTRY)
        rev = self.store.state()["revision"]
        self.confirm(PANTRY[:2])
        with self.assertRaises(HearthError) as ctx:
            self.store.save_dinner_set(rev, PROPOSAL["dinners"], "muse-spark")
        self.assertEqual(ctx.exception.code, "stale_revision")

    def test_invalid_units_refused(self):
        self.confirm(PANTRY)
        bad = [{"id": "x", "name": "X", "ingredients": [{"name": "rice", "quantity": "1", "unit": "cups"}]}]
        with self.assertRaises(HearthError):
            self.store.save_dinner_set(self.store.state()["revision"], bad, "muse-spark")

    def test_new_proposal_invalidates_reviewed_shopping_list(self):
        self.confirm(PANTRY)
        with mock.patch.object(meta_client, "spark_json", return_value=(PROPOSAL, {})):
            out = agents.plan_dinners(self.store, "")
        st = self.store.state()
        preview = self.store.preview_shopping_list(st["revision"], out["plan"]["selected_dinner_ids"],
                                                   st["shopping_revision"])
        other = [{"id": "pasta", "name": "Pasta", "ingredients": [{"name": "pasta", "quantity": "200", "unit": "g"}]}]
        self.store.save_dinner_set(st["revision"], other, "muse-spark")
        with self.assertRaises(HearthError):
            self.store.confirm_shopping_list(preview["id"], preview["content_hash"])


if __name__ == "__main__":
    unittest.main()
