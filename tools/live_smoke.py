"""Live smoke test against the real Meta Model API. Uses a temporary database and fictional pantry data."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents  # noqa: E402
import meta_client  # noqa: E402
from hearth import Store  # noqa: E402

PANTRY = [{"name": "eggs", "quantity": "8", "unit": "each"}, {"name": "jasmine rice", "quantity": "900", "unit": "g"},
          {"name": "frozen peas", "quantity": "400", "unit": "g"}, {"name": "chicken thighs", "quantity": "600", "unit": "g"},
          {"name": "whole milk", "quantity": "1000", "unit": "ml"}, {"name": "baby spinach", "quantity": None, "unit": "g"},
          {"name": "cheddar", "quantity": "200", "unit": "g"}, {"name": "tortillas", "quantity": "8", "unit": "each"}]

step = sys.argv[1] if len(sys.argv) > 1 else "plan"
t0 = time.time()
if step == "hello":
    out, usage = meta_client.spark_json("Reply in JSON.", "Say hi in three words.", "hi", {
        "type": "object", "additionalProperties": False, "required": ["msg"], "properties": {"msg": {"type": "string"}}})
    print(out, usage)
elif step == "plan":
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "h.sqlite3")
        store.confirm(store.preview_inventory(0, PANTRY)["id"])
        result = agents.plan_dinners(store, "It's 5pm, the toddler is hungry. What can we make tonight?")
        print(json.dumps({"spoken": result["spoken_answer"], "assumptions": result["assumptions"],
                          "corrected": result["pantry_flags_corrected"],
                          "dinners": [(d["name"], d.get("why")) for d in result["dinner_set"]["dinners"]],
                          "shopping": [(r["name"], r["status"], r["needed"]) for r in result["plan"]["shopping"]],
                          "usage": result["usage"]}, indent=1))
elif step == "scan":
    path = Path(sys.argv[2])
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    result = agents.scan_image(path.read_bytes(), mime)
    for item in result["items"]:
        print(item["name"], item["quantity"], item["unit"], "sam=%d conf=%s" % (item["sam_matches"], item["sam_confidence"]),
              "CHECK: " + "; ".join(item["reasons"]) if item["needs_check"] else "ok", item["sam_error"] or "")
    print("scene:", result["scene_note"], "| usage:", result["usage"])
print("seconds: %.1f" % (time.time() - t0))
