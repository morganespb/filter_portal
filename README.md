# Hand Filter Portal

Hold both hands up to your webcam and make a frame with your thumbs and index
fingers. The area inside that frame gets a live filter (pencil sketch, purple
tint, thermal, neon edges, cartoon, pixelate, X-ray, invert) while the rest of
the video stays normal.

## Setup

Requires Python 3.9 to 3.12 (MediaPipe wheels may lag behind the newest Python) and a webcam.

```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python filter_portal.py
```

On first run it downloads MediaPipe's hand model (`hand_landmarker.task`, about
7 MB) next to the script. On macOS, allow camera access for your terminal/IDE
when prompted.

Options: `--camera 1` to pick another webcam, `--width 1920 --height 1080`
for resolution, `--no-mirror` to turn off the selfie flip.

## Controls

| Action | What it does |
|---|---|
| Thumbs + index fingers of both hands | Draws the portal; filter applies inside |
| Bunch all 5 fingertips of one hand together | Next filter (open your hand before switching again) |
| `n` / Space, `p` | Next / previous filter |
| `s` | Split mode: two filters, divided along the portal's diagonal (the sketch + purple look) |
| `m` | Mirror on/off |
| `h` | Show/hide help |
| `q` / Esc | Quit |

## Adding your own filter

Write a function that takes a BGR image and returns one of the same size, then
add it to the `FILTERS` list in `filter_portal.py`:

```python
def f_red(img):
    out = img.copy(); out[..., :2] //= 3
    return out

FILTERS.append(("Red", f_red))
```

## Troubleshooting

- "Could not open camera": try `--camera 1`, close other apps using the camera.
- Portal flickers: improve lighting and keep both hands fully in frame.
- `module 'mediapipe' has no attribute 'solutions'`: not used here; this
  program uses the newer MediaPipe Tasks API, so it works on current versions.
