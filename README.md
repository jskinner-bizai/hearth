# Hearth

Hearth keeps track of what's actually in the house, so the 5pm "what can we make for dinner?" question takes a minute instead of twenty.

You pan your phone across the fridge or take a photo. Meta's models work out what's there and how many of each. You fix anything they got wrong, confirm, and then ask out loud what to make. Hearth proposes two dinners from what you have, works out exactly what to pick up, and only saves a shopping list when you say so.

Built for the Meta Global AI Developer Hackathon (Indie track, Build for Your World).

## Why I built it

I have a toddler and a full-time job. Most weeknights dinner gets decided by whoever opens the fridge first, usually after we've already bought something we had. The hard part isn't recipes. It's knowing what you actually have, in what amounts, without standing in front of the fridge taking notes.

## How it uses Meta's models

| Step | Model | What it does |
|---|---|---|
| Pan the fridge | Muse Spark 1.3 (video input) | Watches the clip and lists each food it saw |
| Count each thing once | SAM 3.1 video tracking | Tracks every yogurt cup, can and pepper across the frames with a stable ID, so an item that shows up in 60 frames is still one item |
| Or take a photo | Muse Spark 1.3 + SAM 3.1 box prompts | Spark boxes every unit it can see; SAM confirms each box and returns a confidence score |
| Ask by voice | Muse Voice Transcribe | Transcribes the question with keyword biasing from your own pantry names, so "gochujang" or your kid's yogurt brand comes through right |
| Plan dinner | Muse Spark 1.3 (constrained JSON) | Proposes two dinners from the confirmed pantry, with amounts |

The models draft and propose. Hearth decides what gets saved:

- Low SAM confidence, Spark and SAM disagreeing on a count, or an item Spark only half saw all land in a "please check" list instead of going straight into the pantry.
- Nothing reaches the saved pantry without a review and an explicit confirm. Every confirm writes a receipt.
- Spark's dinner proposal is checked against the saved pantry. Hearth recomputes every shopping amount itself and says out loud where it disagrees with the model (for example, when the pantry doesn't record how much spinach you have).
- A proposal is tied to the pantry revision it was planned against. If the pantry changes, the old plan and any shopping list based on it are refused.

The spoken answer uses the browser's speech synthesis. Meta's API doesn't offer text to speech.

## Run it

Python 3.12 and Pillow.

```
pip install -r requirements.txt
set META_API_KEY=...            (PowerShell: $env:META_API_KEY="...")
python server.py --port 8140
```

Open http://127.0.0.1:8140. To serve it behind HTTPS on a public hostname, set `HEARTH_PUBLIC_HOST` to that hostname and bind with `--bind 0.0.0.0`. Model calls are rate limited per minute (`HEARTH_AI_PER_MIN`, default 12) so a public demo can't run up the bill.

Tests (no network, no API key needed):

```
python -m unittest discover -s tests -t .
```

`tools/live_smoke.py` and `tools/video_smoke.py` run the real model calls against your own photo or clip.

## What's in here

| File | Purpose |
|---|---|
| `meta_client.py` | Small stdlib client for Muse Spark, SAM 3.1 (image and video, box and point prompts, streamed output parser) and Muse Voice Transcribe |
| `scan_v2.py` | Photo scan: Spark boxes, SAM confirmation, review flags |
| `scan_video.py` | Pan scan: Spark on video, SAM tracking with stable IDs |
| `agents.py` | Dinner planning, Hearth's own check of the plan, voice transcription |
| `hearth.py` | Pantry state: revisions, expiring previews, atomic confirms, receipts, shopping lists |
| `server.py` | HTTP server with origin checks, JSON validation and size limits |
| `static/index.html` | The whole UI in one file |

## What I'd do next

Expiry dates read off labels, a weekly view so leftovers get used before they go bad, and sharing the pantry between two phones so both parents see the same list.

## Notes

All of this was written during the hackathon, starting October 26, 2026. The demo household and the fridge in the video are mine; no other family data is in the repo.

MIT licensed.
