import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scan_video  # noqa: E402

t = time.time()
out = scan_video.scan_video(Path(sys.argv[1]).read_bytes(), "video/mp4")
for i in out["items"]:
    print(json.dumps({k: i[k] for k in ("name", "quantity", "unit", "counted_by", "sam_matches", "frames_tracked", "reasons")}))
print("scene:", out["scene_note"])
print("seconds: %.1f" % (time.time() - t))
