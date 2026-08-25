# Pose-Based Fall Detection (YOLOv8-Pose)

Detect when a person **falls** from a camera feed, while deliberately **not**
raising false alarms for normal low postures such as **praying** (prostration /
sujood) or **kneeling / sitting on the floor**.

## The problem

A fall is a critical safety event (elderly care, hospitals, workplaces). The
naive approach, training a plain object detector on "fallen person" boxes, is
fragile: a person kneeling, praying, crouching, or bending to pick something up
often produces the same short, wide bounding box a fallen person does, so the
detector fires constant false alarms.

This project instead reasons about **body geometry** using pose estimation, so
we can tell a real fall (body horizontal on the ground) apart from a controlled
low posture (torso still upright, body still folded and vertically stacked).

## Why pose-based

We use **YOLOv8-Pose** (`yolov8n-pose.pt` or `yolov8s-pose.pt`), which returns
the 17 COCO body key-points per person (shoulders, hips, knees, ankles, etc.).
From those key-points we compute meaningful features that a bounding box alone
cannot capture:

| Feature | Meaning |
|---|---|
| `torso_angle` | Angle of the spine (hip to shoulder) from vertical. ~0 = upright, ~90 = lying flat. |
| `bbox_ratio` | Person box width / height. > 1 means wider than tall. |
| `spread_ratio` | Horizontal vs vertical spread of the key-points. |
| `knee_angle` | How bent the knees are (folded legs are typical of kneeling/prayer). |
| vertical stacking | Whether the hips sit clearly above the ankles (feet tucked under the body). |

## How a fall is decided

A **FALL** is flagged only when the body is genuinely horizontal:

```
torso is close to horizontal  AND  the body is wider than tall  AND  not vertically stacked
```

**Praying and kneeling are rejected on purpose**, because in those postures:

- the torso stays upright (or the body stays folded), so it is not "horizontal", and
- the hips remain clearly above the ankles (feet tucked under), so the body is
  still *vertically stacked* and the box stays taller-than-wide.

They therefore fail the fall test and are shown as `No Fall` (or, when detailed
posture labels are enabled, as `Kneeling/Praying`). See "Camera placement
matters" below for why only the coarse label is trustworthy on some cameras.

### Temporal confirmation

For video and webcam, each person is tracked with a stable ID and a fall is only
declared after the condition holds for several consecutive frames
(`FALL_CONFIRM_FRAMES`). This removes single-frame noise and ignores brief bends
(tying a shoe, picking something up).

## Project structure

```
Fall_detection/
  falldetection.py     # main script (detection + posture logic)
  requirements.txt
  README.md
  models/              # YOLOv8-pose weights (auto-downloaded here on first run)
  data/                # put your test images / videos here
  output/              # annotated results are written here
```

## Setup

Uses the shared D-Fire virtual environment (Python 3.12). From the `D-Fire`
folder:

```bash
source venv/bin/activate
pip install -r Fall_detection/requirements.txt
```

`ultralytics`, `torch`, `opencv-python` and `numpy` are already present in the
project venv, so this is usually a no-op. On first run the chosen pose model
(`yolov8n-pose.pt` by default) is downloaded automatically into `models/`.

## Usage

Interactive menu (webcam / video / image / folder):

```bash
python falldetection.py
```

Command line:

```bash
python falldetection.py --source 0                    # live webcam
python falldetection.py --source data/clip.mp4         # video file
python falldetection.py --source data/photo.jpg        # single image
python falldetection.py --source data/                 # folder of images
python falldetection.py --model s                      # use yolov8s-pose (more accurate)
python falldetection.py --source data/clip.mp4 --no-show   # headless
```

Annotated output (skeleton + label, plus a red `FALL DETECTED` banner when a
fall is confirmed) is saved to `output/`.

### Preset input paths

If you would rather not pass `--source` each time, set the two constants near
the top of `falldetection.py` and just run `python falldetection.py`:

```python
VIDEO_PATH = os.path.join(DATA_DIR, "test_video.mp4")   # e.g. data/test_video.mp4
IMAGE_PATH = os.path.join(DATA_DIR, "test_image.jpg")   # e.g. data/test_image.jpg
```

In the interactive menu, options 2 (video) and 3 (image) show these paths as the
default in brackets, press Enter to use them or type another path to override.

## Tuning

All thresholds live in the `CONFIG` block at the top of `falldetection.py`.
Camera angle matters a lot, so adjust these for your scene:

- `TORSO_HORIZONTAL_DEG` (default `55`): lower it if real falls are missed,
  raise it if bending is misread as a fall.
- `FALL_BBOX_RATIO` / `FALL_SPREAD_RATIO`: how "wide" the body must be.
- `KNEE_BENT_DEG`: how bent the knees must be to count as kneeling/prayer.
- `FALL_CONFIRM_FRAMES`: higher = fewer false alarms but slower to react.

## Labels shown on screen

By default (`SHOW_POSTURE_LABELS = False`) only the reliable state is shown:

| Label | Meaning | Color |
|---|---|---|
| `No Fall` | Person present, upright or in a normal low posture (incl. prayer) | green |
| `Falling?` | Looks like a fall, awaiting temporal confirmation | orange |
| `FALL` | Confirmed fall | red |

If you set `SHOW_POSTURE_LABELS = True`, the detailed posture name is shown
instead (`Standing`, `Sitting/Bending`, `Kneeling/Praying`). See the camera note
below before enabling it.

## Camera placement matters (important)

The fine posture names are derived from **2D image geometry**, which only matches
real body orientation when the camera is roughly at **eye level / side-on**.

From an **elevated, angled-down or top-down** camera the geometry no longer maps
to reality: a standing person's legs look bent (foreshortened), bowing (ruku)
still looks vertical, and prostration (sujood) does not look "flat". In that case
the posture *names* can be wrong even though **fall vs no-fall stays correct** (a
real fall spreads the body flat across the floor, which no prayer pose does).

Guidance:
- Overhead / angled camera -> keep `SHOW_POSTURE_LABELS = False` (honest
  `No Fall` / `FALL`). This is the reliable mode and still fully meets the goal
  of not false-alarming on prayer or kneeling.
- Eye-level / side camera -> you may set `SHOW_POSTURE_LABELS = True` for the
  detailed posture label.
- Accurate per-pose recognition (qiyam / ruku / sujood / attahiyat) from an
  overhead angle needs a trained pose-sequence classifier, not geometric rules.

## Notes and limitations

- Accuracy depends on camera height/angle; a near top-down view makes torso
  angle less reliable, so tune thresholds accordingly.
- Heavy occlusion or very small/distant people reduce key-point confidence and
  can cause missed detections.
- The rules are geometric and interpretable by design; for higher accuracy in a
  specific deployment you can log the printed metrics and fit thresholds (or a
  small classifier) to your own footage.
