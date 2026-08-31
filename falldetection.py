"""
Pose-Based Fall Detection using YOLOv8-Pose
============================================

Problem
-------
Detect when a person has FALLEN (an emergency event) from a camera feed, while
NOT raising false alarms for normal low-postures such as PRAYING (prostration /
sujood) or KNEELING / sitting on the floor.

Why pose-based (not a plain object detector)?
---------------------------------------------
A bounding-box-only "fall" detector learns appearance and is easily fooled: a
person kneeling to pray, bending to pick something up, or crouching can produce
the same "short/wide" box a fallen person does. By using body key-points
(skeleton) we can reason about *geometry* -- the orientation of the torso and
the layout of the limbs -- which separates a real fall (body horizontal on the
ground) from a controlled low posture (torso still upright / body still folded
and vertically stacked).

Model
-----
YOLOv8-Pose (nano `yolov8n-pose.pt` or small `yolov8s-pose.pt`). It outputs the
17 COCO key-points per person:

    0 nose        5 left_shoulder   11 left_hip     15 left_ankle
    1 left_eye    6 right_shoulder  12 right_hip     16 right_ankle
    2 right_eye   7 left_elbow      13 left_knee
    3 left_ear    8 right_elbow     14 right_knee
    4 right_ear   9 left_wrist      10 right_wrist

Fall logic (single frame)
-------------------------
For each detected person we compute:
  * torso_angle     : angle of the hip->shoulder (spine) vector from the
                      vertical axis.  ~0 = upright, ~90 = lying flat.
  * bbox_ratio      : width / height of the person box.  > 1 = wider than tall.
  * body_spread     : horizontal spread vs vertical spread of all key-points.
  * knee_bend       : how bent the knees are (hip-knee-ankle angle).

A FALL is flagged only when the body is genuinely horizontal:
      torso is close to horizontal  AND  the box/keypoints are wider than tall.

Praying / kneeling are explicitly rejected because in those postures the body
stays vertically stacked (hips above knees/ankles) so the box is taller-than-
wide and/or the torso is not horizontal -- they fail the fall test on purpose.

Temporal confirmation
----------------------
A single frame can be noisy, so for video/webcam we track each person (YOLO
tracker) and only declare "FALL" after the fall condition holds for a few
consecutive frames.  This also ignores brief bends (tying a shoe, picking
something up).

Usage
-----
    python falldetection.py
        -> interactive menu (webcam / video / image / folder)

    python falldetection.py --source 0                 # webcam
    python falldetection.py --source path/to/video.mp4  # video file
    python falldetection.py --source path/to/image.jpg  # single image
    python falldetection.py --source data/              # folder of images
    python falldetection.py --model s                   # use yolov8s-pose
    python falldetection.py --no-show                   # headless (no window)
"""

import os
import sys
import argparse
import shutil
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for _d in (MODELS_DIR, DATA_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# INPUT PATHS  (edit these, then just run `python falldetection.py`)
# ----------------------------------------------------------------------------
# Set the video or image you want to test. They can point anywhere, but placing
# your files in the data/ folder keeps the project self-contained.
# Give either one -- the interactive menu offers both and uses these as the
# default when you press Enter without typing a path.
VIDEO_PATH = os.path.join(DATA_DIR, "fall.mp4")   # e.g. data/test_video.mp4
IMAGE_PATH = os.path.join(DATA_DIR, "MaleRest.png")   # e.g. data/test_image.jpg

# ----------------------------------------------------------------------------
# CONFIG  (tune these thresholds for your camera angle / scene)
# ----------------------------------------------------------------------------
# Which pose model to use by default: "n" (fast) or "s" (more accurate).
DEFAULT_MODEL_SIZE = "n"

# Detection / key-point confidence.
PERSON_CONF = 0.25          # min person detection confidence
KPT_CONF = 0.30             # min confidence for a key-point to be trusted
IMG_SIZE = 1280             # inference resolution (higher = catches small/distant
                            # people in crowded scenes, but slower). Lower to 960
                            # or 640 for speed if the scene has few, large people.

# --- Fall geometry thresholds -----------------------------------------------
# Torso (spine) angle from the vertical axis, in degrees.
#   0  = perfectly upright (standing / kneeling / praying upright)
#   90 = perfectly horizontal (lying on the ground)
TORSO_HORIZONTAL_DEG = 55.0     # torso considered "horizontal" above this
TORSO_UPRIGHT_DEG = 35.0        # torso considered "upright" below this

# Person bounding box width/height ratio. A fallen person is wider than tall.
FALL_BBOX_RATIO = 1.10

# Horizontal-vs-vertical spread of the key-points. >1 means body lies along x.
FALL_SPREAD_RATIO = 1.10

# Knees strongly bent (hip-knee-ankle angle below this) => folded legs, which is
# typical of kneeling / prayer, NOT of a person lying stretched out.
KNEE_BENT_DEG = 130.0

# --- MOTION-BASED fall detection (video / webcam only) ----------------------
# A FALL is the *event* of a person who was UPRIGHT rapidly ending up on the
# GROUND. This is what separates a real fall from someone who is simply lying
# down, sitting, or praying: those people are never seen making a fast
# upright -> ground transition (they are already low, or they lower themselves
# slowly). Everything below is expressed per tracked person.
#
# "Upright" and "grounded" are read from the bounding-box aspect ratio (w/h),
# which is robust across camera angles: standing people are taller than wide,
# people on the floor are wider than tall.
UPRIGHT_AR = 0.75           # w/h <= this  => person is upright (taller than wide)
GROUND_AR = 1.05            # w/h >= this  => person is on the ground (wider)
FALL_SPAN_SEC = 0.9         # the upright->ground transition must complete within
                            # this many seconds to count as a *fall* (fast).
                            # A slow, deliberate lie-down takes longer and is
                            # therefore treated as normal.
GROUND_CONFIRM = 3          # grounded for at least this many frames before alarm
FALL_HOLD_SEC = 3.0         # keep the FALL label this long after the event
                            # (cleared early if the person stands back up)

# COCO skeleton links for drawing.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head to shoulders
]

# Colors (BGR)
COLOR_OK = (0, 200, 0)         # normal / no fall
COLOR_WATCH = (0, 200, 255)    # low posture but not a fall (kneeling/praying/sitting)
COLOR_FALL = (0, 0, 255)       # confirmed fall

# Fine posture names (Standing / Kneeling-Praying / Sitting-Bending) are only
# reliable from an eye-level or side view. From an elevated / angled-down camera
# the 2D geometry no longer reflects real body orientation, so those names can be
# wrong even though FALL vs NO-FALL stays correct. Keep this False for overhead /
# angled cameras (honest label: "No Fall" / "FALL"); set True only when the
# camera is roughly at eye level and you want the detailed posture label.
SHOW_POSTURE_LABELS = False


# ----------------------------------------------------------------------------
# GEOMETRY HELPERS
# ----------------------------------------------------------------------------
def _valid(pt_conf):
    """A key-point is usable if its confidence clears the threshold."""
    return pt_conf is not None and pt_conf >= KPT_CONF


def _midpoint(kpts, confs, i, j):
    """Mean of two key-points, using whichever of the pair is confident.

    Returns None if neither key-point is reliable.
    """
    a_ok, b_ok = _valid(confs[i]), _valid(confs[j])
    if a_ok and b_ok:
        return (kpts[i] + kpts[j]) / 2.0
    if a_ok:
        return kpts[i].copy()
    if b_ok:
        return kpts[j].copy()
    return None


def _angle_from_vertical(p_low, p_high):
    """Angle (deg) of the vector p_low->p_high measured from the vertical axis.

    0   -> the two points are vertically aligned (upright body segment)
    90  -> the two points are horizontally aligned (segment lying flat)
    """
    if p_low is None or p_high is None:
        return None
    dx = float(p_high[0] - p_low[0])
    dy = float(p_high[1] - p_low[1])
    # atan2(|dx|, |dy|): dominated by dy -> near 0, dominated by dx -> near 90
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _joint_angle(a, b, c):
    """Interior angle (deg) at joint b formed by segments b->a and b->c."""
    if a is None or b is None or c is None:
        return None
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ----------------------------------------------------------------------------
# POSTURE ANALYSIS  (the heart of the false-positive suppression)
# ----------------------------------------------------------------------------
def analyze_posture(kpts, confs, bbox):
    """Classify a single person's posture from key-points + bounding box.

    Parameters
    ----------
    kpts  : (17, 2) array of key-point (x, y) pixel coordinates
    confs : (17,) array of key-point confidences
    bbox  : (x1, y1, x2, y2) person box

    Returns
    -------
    dict with:
        label     : "Standing" | "Sitting/Bending" | "Kneeling/Praying" | "FALL"
        is_fall   : bool (raw, single-frame decision)
        metrics   : dict of the computed geometric features (for overlay/debug)
    """
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    bbox_ratio = box_w / box_h

    shoulder = _midpoint(kpts, confs, 5, 6)
    hip = _midpoint(kpts, confs, 11, 12)
    knee = _midpoint(kpts, confs, 13, 14)
    ankle = _midpoint(kpts, confs, 15, 16)
    head = kpts[0] if _valid(confs[0]) else shoulder

    # Torso orientation: the single most important cue.
    torso_angle = _angle_from_vertical(hip, shoulder)

    # Leg orientation and knee bend (help separate kneeling/prayer from lying).
    leg_angle = _angle_from_vertical(ankle, hip)
    knee_l = _joint_angle(kpts[11], kpts[13], kpts[15]) if all(
        _valid(confs[k]) for k in (11, 13, 15)) else None
    knee_r = _joint_angle(kpts[12], kpts[14], kpts[16]) if all(
        _valid(confs[k]) for k in (12, 14, 16)) else None
    knee_angles = [k for k in (knee_l, knee_r) if k is not None]
    knee_angle = min(knee_angles) if knee_angles else None

    # Horizontal vs vertical spread of the confident key-points.
    valid_pts = np.array([kpts[i] for i in range(len(kpts)) if _valid(confs[i])])
    if len(valid_pts) >= 2:
        x_span = float(valid_pts[:, 0].max() - valid_pts[:, 0].min())
        y_span = float(valid_pts[:, 1].max() - valid_pts[:, 1].min())
        spread_ratio = x_span / (y_span + 1e-6)
    else:
        spread_ratio = bbox_ratio

    metrics = {
        "torso_angle": torso_angle,
        "leg_angle": leg_angle,
        "knee_angle": knee_angle,
        "bbox_ratio": bbox_ratio,
        "spread_ratio": spread_ratio,
    }

    # --- Boolean cues --------------------------------------------------------
    torso_horizontal = (torso_angle is not None) and (torso_angle >= TORSO_HORIZONTAL_DEG)
    torso_upright = (torso_angle is not None) and (torso_angle <= TORSO_UPRIGHT_DEG)
    body_wide = (bbox_ratio >= FALL_BBOX_RATIO) or (spread_ratio >= FALL_SPREAD_RATIO)
    knees_folded = (knee_angle is not None) and (knee_angle < KNEE_BENT_DEG)

    # Vertical stacking check: in kneeling/prayer the hips stay clearly above the
    # ankles (feet tucked under the body). When lying down, hips and ankles are
    # at roughly the same height. We measure the hip->ankle vertical drop
    # relative to the person's overall height.
    stacked = False
    if hip is not None and ankle is not None:
        vertical_drop = float(ankle[1] - hip[1])   # +ve => ankle below hip
        stacked = vertical_drop > 0.20 * box_h

    # --- Decision ------------------------------------------------------------
    # 1) Real fall: torso horizontal AND the body is spread out horizontally.
    #    Kneeling/praying never satisfy "torso horizontal + wide body" together
    #    because the body stays folded and vertically stacked.
    if torso_horizontal and body_wide and not stacked:
        label, is_fall = "FALL", True

    # 2) Kneeling / praying: legs are clearly folded AND the body is still
    #    vertically stacked (hips above tucked feet). Explicitly NOT a fall.
    elif knees_folded and stacked and not body_wide:
        label, is_fall = "Kneeling/Praying", False

    # 3) Upright and tall (legs not folded) -> standing / walking.
    elif torso_upright and not body_wide and not knees_folded:
        label, is_fall = "Standing", False

    # 4) Everything else (bending forward, sitting, transitional) -> not a fall.
    else:
        label, is_fall = "Sitting/Bending", False

    return {"label": label, "is_fall": is_fall, "metrics": metrics}


# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
def draw_person(frame, kpts, confs, bbox, label, color, track_id=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # skeleton
    for a, b in SKELETON:
        if _valid(confs[a]) and _valid(confs[b]):
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for i in range(len(kpts)):
        if _valid(confs[i]):
            cv2.circle(frame, (int(kpts[i][0]), int(kpts[i][1])), 3, color, -1)

    tag = f"{label}" if track_id is None else f"ID{track_id} {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_alert_banner(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def decide_display(info, confirmed_fall):
    """Map a posture result to the on-screen (label, color).

    When SHOW_POSTURE_LABELS is False we only assert what is reliable from any
    camera angle: FALL vs "No Fall" (plus a transient "Falling?" while a possible
    fall is still being confirmed). When True we show the detailed posture name,
    which is only trustworthy from an eye-level / side view.
    """
    if confirmed_fall:
        return "FALL", COLOR_FALL

    if SHOW_POSTURE_LABELS:
        if info["label"] == "FALL":
            return "Falling?", COLOR_WATCH
        if info["label"] in ("Kneeling/Praying", "Sitting/Bending"):
            return info["label"], COLOR_WATCH
        return info["label"], COLOR_OK

    # Honest coarse mode.
    return "No Fall", COLOR_OK


# ----------------------------------------------------------------------------
# MOTION-BASED FALL MONITOR  (per tracked person, across frames)
# ----------------------------------------------------------------------------
class FallMonitor:
    """Detects the *event* of falling: an UPRIGHT person who rapidly ends up on
    the GROUND. People who are already lying, sitting, or praying, or who lower
    themselves slowly, never trigger it.

    For each tracked person we keep a short rolling history of the bounding-box
    aspect ratio (w/h) with frame indices. A fall fires when the current state
    is "grounded" and, within the last FALL_SPAN frames, the same person was
    clearly "upright" (i.e. a fast upright -> ground transition).
    """

    def __init__(self, fps):
        fps = fps if fps and fps > 0 else 30
        self.span = max(4, int(FALL_SPAN_SEC * fps))     # transition window
        self.hold = max(1, int(FALL_HOLD_SEC * fps))     # alarm hold time
        self.hist = defaultdict(lambda: deque(maxlen=self.span + 2))
        self.ground_streak = defaultdict(int)            # consecutive grounded frames
        self.fall_until = {}                             # tid -> frame to hold FALL until

    def update(self, tid, bbox, frame_idx):
        """Feed one detection for a tracked person; return (is_fall, is_grounded)."""
        if tid is None:
            # Without a stable track id we cannot reason about motion safely.
            return False, False

        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        ar = w / h
        self.hist[tid].append((frame_idx, ar))

        grounded = ar >= GROUND_AR
        upright = ar <= UPRIGHT_AR

        self.ground_streak[tid] = self.ground_streak[tid] + 1 if grounded else 0

        # Standing back up clears any active fall.
        if upright:
            self.fall_until.pop(tid, None)

        # Fall event: grounded now (confirmed for a few frames) AND was upright
        # recently within the transition window -> a fast collapse.
        if grounded and self.ground_streak[tid] >= GROUND_CONFIRM:
            recent_upright = any(
                a <= UPRIGHT_AR and (frame_idx - f) <= self.span
                for (f, a) in self.hist[tid]
            )
            if recent_upright:
                self.fall_until[tid] = frame_idx + self.hold

        is_fall = self.fall_until.get(tid, -1) >= frame_idx
        return is_fall, grounded


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
def load_model(size):
    """Load YOLOv8-pose, keeping weights inside the models/ folder."""
    size = size.lower().strip()
    if size not in ("n", "s"):
        print(f"Unknown model size '{size}', falling back to 'n'.")
        size = "n"
    model_name = f"yolov8{size}-pose.pt"
    weights = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(weights):
        print(f"'{model_name}' not found in models/. Downloading official weights...")
        # YOLO(model_name) auto-downloads into the current working dir; move it
        # into models/ so the project stays self-contained.
        _ = YOLO(model_name)
        if os.path.exists(model_name) and os.path.abspath(model_name) != os.path.abspath(weights):
            shutil.move(model_name, weights)

    print(f"Loading pose model: {weights}")
    model = YOLO(weights)
    print("Model loaded.")
    return model


# ----------------------------------------------------------------------------
# EXTRACT PER-PERSON DATA FROM A YOLO RESULT
# ----------------------------------------------------------------------------
def extract_people(result):
    """Yield (track_id, kpts(17,2), confs(17,), bbox(4,)) for each person."""
    people = []
    if result.keypoints is None or result.boxes is None:
        return people

    kpts_xy = result.keypoints.xy.cpu().numpy()            # (N, 17, 2)
    kpts_cf = (result.keypoints.conf.cpu().numpy()
               if result.keypoints.conf is not None
               else np.ones(kpts_xy.shape[:2]))            # (N, 17)
    boxes = result.boxes.xyxy.cpu().numpy()                # (N, 4)
    box_cf = result.boxes.conf.cpu().numpy()               # (N,)
    ids = (result.boxes.id.cpu().numpy().astype(int)
           if result.boxes.id is not None else [None] * len(boxes))

    for i in range(len(boxes)):
        if box_cf[i] < PERSON_CONF:
            continue
        people.append((ids[i], kpts_xy[i], kpts_cf[i], boxes[i]))
    return people


# ----------------------------------------------------------------------------
# RUN ON A STREAM (video file or webcam)
# ----------------------------------------------------------------------------
def run_stream(model, source, show=True, out_name="fall_detection_output.mp4"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(OUTPUT_DIR, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Motion-based fall detector (per tracked person, across frames).
    monitor = FallMonitor(fps)

    print("Processing... press 'q' in the window to stop.")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Tracking gives stable IDs so motion can be reasoned per-person.
        results = model.track(
            frame, persist=True, conf=PERSON_CONF, imgsz=IMG_SIZE,
            verbose=False, classes=[0]
        )
        result = results[0]

        any_fall = False
        for track_id, kpts, confs, bbox in extract_people(result):
            info = analyze_posture(kpts, confs, bbox)
            # FALL is decided by MOTION (upright -> ground fast), not by a static
            # pose. This keeps lying / sitting / praying people as "No Fall".
            is_fall, _grounded = monitor.update(track_id, bbox, frame_idx)
            if is_fall:
                any_fall = True

            label, color = decide_display(info, is_fall)
            draw_person(frame, kpts, confs, bbox, label, color, track_id)

        if any_fall:
            draw_alert_banner(frame, "FALL DETECTED")

        writer.write(frame)
        if show:
            disp = cv2.resize(frame, (960, 540)) if w > 960 else frame
            cv2.imshow("Pose-Based Fall Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done. Saved: {out_path}")


# ----------------------------------------------------------------------------
# RUN ON A SINGLE IMAGE
# ----------------------------------------------------------------------------
def run_image(model, img_path, show=True):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: could not read image '{img_path}'.")
        return

    result = model(frame, conf=PERSON_CONF, imgsz=IMG_SIZE, verbose=False, classes=[0])[0]

    # NOTE: a fall is a MOTION event (a person going down). It cannot be judged
    # from a single still image, so here we only report posture and never raise a
    # FALL. Use a video / webcam for actual fall detection.
    for _tid, kpts, confs, bbox in extract_people(result):
        info = analyze_posture(kpts, confs, bbox)
        label, color = decide_display(info, False)
        m = info["metrics"]
        ta = f"{m['torso_angle']:.0f}" if m["torso_angle"] is not None else "-"
        # print keeps the detailed posture + metrics for tuning/debugging
        print(f"  person: shown='{label}'  detail={info['label']:16s} "
              f"torso_angle={ta}  bbox_ratio={m['bbox_ratio']:.2f}  "
              f"spread={m['spread_ratio']:.2f}")
        draw_person(frame, kpts, confs, bbox, label, color)

    out_path = os.path.join(OUTPUT_DIR, "output_" + os.path.basename(img_path))
    cv2.imwrite(out_path, frame)
    print(f"Saved: {out_path}")

    if show:
        cv2.imshow("Pose-Based Fall Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_folder(model, folder, show=False):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    if not imgs:
        print(f"No images found in '{folder}'.")
        return
    print(f"Running on {len(imgs)} image(s) in {folder} ...")
    for name in imgs:
        print(f"\n{name}")
        run_image(model, os.path.join(folder, name), show=show)


# ----------------------------------------------------------------------------
# CLI / INTERACTIVE ENTRY POINT
# ----------------------------------------------------------------------------
def interactive(model):
    print("\nSelect input source:")
    print("  1) Live webcam")
    print("  2) Video file")
    print("  3) Single image")
    print("  4) Folder of images")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        run_stream(model, 0, show=True)
    elif choice == "2":
        path = input(f"Enter video path [{VIDEO_PATH}]: ").strip() or VIDEO_PATH
        if not os.path.exists(path):
            print(f"Video not found: {path}")
            return
        run_stream(model, path, show=True,
                   out_name=os.path.splitext(os.path.basename(path))[0] + "_fall.mp4")
    elif choice == "3":
        path = input(f"Enter image path [{IMAGE_PATH}]: ").strip() or IMAGE_PATH
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return
        run_image(model, path, show=True)
    elif choice == "4":
        path = input(f"Enter folder path [{DATA_DIR}]: ").strip() or DATA_DIR
        if not os.path.isdir(path):
            print(f"Folder not found: {path}")
            return
        run_folder(model, path, show=False)
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Pose-based fall detection (YOLOv8-Pose)")
    parser.add_argument("--source", help="0 for webcam, or path to video/image/folder")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, choices=["n", "s"],
                        help="YOLOv8-pose size: n (fast) or s (accurate)")
    parser.add_argument("--no-show", action="store_true", help="run headless (no window)")
    args = parser.parse_args()

    model = load_model(args.model)
    show = not args.no_show

    if args.source is None:
        interactive(model)
        return

    src = args.source
    if src == "0" or src == "1":
        run_stream(model, int(src), show=show)
    elif os.path.isdir(src):
        run_folder(model, src, show=show)
    elif src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_image(model, src, show=show)
    else:
        run_stream(model, src, show=show,
                   out_name=os.path.splitext(os.path.basename(src))[0] + "_fall.mp4")


if __name__ == "__main__":
    main()
