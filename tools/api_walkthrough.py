"""Drive the same HTTP endpoints the page uses: scan a photo, review the draft, confirm it.

Usage: python tools/api_walkthrough.py http://127.0.0.1:8141 path/to/fridge.jpg
"""
import json
import sys
import urllib.request

base, photo = sys.argv[1].rstrip("/"), sys.argv[2]


def call(path, body, ctype="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": ctype, "Origin": base})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


scan = call("/api/ai/scan", open(photo, "rb").read(), "image/jpeg")
print("scan:", len(scan["items"]), "items,", sum(i["needs_check"] for i in scan["items"]), "to check")
for i in scan["items"]:
    print("  ", i["name"], i["quantity"], i["unit"], "CHECK " + "; ".join(i["reasons"]) if i["needs_check"] else "ok")
state = json.loads(urllib.request.urlopen(base + "/api/state").read())
items = [{"name": i["name"], "quantity": i["quantity"], "unit": i["unit"]} for i in scan["items"]]
preview = call("/api/preview", {"expected_revision": state["revision"], "items": items})["preview"]
receipt = call("/api/confirm", {"preview_id": preview["id"]})
print("confirmed revision", receipt["receipt"]["to_revision"], "receipt", receipt["receipt"]["id"][:8])
