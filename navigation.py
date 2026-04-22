import math
import time
import threading
import queue
import pyttsx3
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Camera parameters
# ─────────────────────────────────────────────
IMAGE_W, IMAGE_H = 512, 512
CX, CY = IMAGE_W // 2, IMAGE_H // 2
V_FOV_RAD = math.radians(170)
H_CM = 145  # camera height in cm


# ─────────────────────────────────────────────
# Object classification maps
# ─────────────────────────────────────────────

# How dangerous/urgent is each label? Lower = higher priority.
OBJECT_PRIORITY = {
    "pothole":          1,
    "drain":            1,
    "manhole":          1,
    "road_block":       2,
    "road_hump":        2,
    "traffic_light":    3,
    "light_red":        3,
    "light_yellow":     3,
    "light_green":      3,
    "light_ped_stop":   3,
    "light_ped_walk":   3,
    "person":           4,
    "rider":            4,
    "dog":              4,
    "cow":              4,
    "buffalo":          4,
    "mule":             4,
    "car":              5,
    "motorcycle":       5,
    "autorickshaw":     5,
    "bus":              5,
    "truck":            5,
    "bicycle":          5,
    "e-rickshaw":       5,
    "rickshaw":         5,
    "hand_cart":        6,
    "traffic_sign":     6,
    "pavement_marking": 7,
    "hawker_stall":     7,
    "trash_bin":        7,
    "temp_shelter":     7,
    "bench":            7,
    "fire_hydrant":     7,
    "garbage":          7,
    "bush":             7,
    "pole":             8,
    "electric_pole":    8,
    "wall":             8,
    "building":         9,
    "tree":             9,
    "gate_open":        9,
    "gate_closed":      9,
    "billboard":       10,
}

# Human-friendly names for labels
FRIENDLY_NAMES = {
    "pothole":          "pothole",
    "drain":            "open drain",
    "manhole":          "manhole",
    "road_block":       "road block",
    "road_hump":        "speed bump",
    "traffic_light":    "traffic signal",
    "light_red":        "red signal",
    "light_yellow":     "yellow signal",
    "light_green":      "green signal",
    "light_ped_stop":   "pedestrian stop signal",
    "light_ped_walk":   "pedestrian walk signal",
    "person":           "person",
    "rider":            "rider",
    "dog":              "dog",
    "cow":              "cow",
    "buffalo":          "buffalo",
    "mule":             "mule",
    "car":              "car",
    "motorcycle":       "motorcycle",
    "autorickshaw":     "auto-rickshaw",
    "bus":              "bus",
    "truck":            "truck",
    "bicycle":          "bicycle",
    "e-rickshaw":       "e-rickshaw",
    "rickshaw":         "rickshaw",
    "hand_cart":        "hand cart",
    "traffic_sign":     "traffic sign",
    "pavement_marking": "pavement marking",
    "hawker_stall":     "hawker stall",
    "trash_bin":        "dustbin",
    "temp_shelter":     "temporary shelter",
    "bench":            "bench",
    "fire_hydrant":     "fire hydrant",
    "garbage":          "garbage",
    "bush":             "bush",
    "pole":             "pole",
    "electric_pole":    "electric pole",
    "wall":             "wall",
    "building":         "building",
    "tree":             "tree",
    "gate_open":        "open gate",
    "gate_closed":      "closed gate",
    "billboard":        "billboard",
}

# Actions to suggest: obstacles on left → go right, and vice versa.
# Objects that are ground hazards (approach from center) → stop/slow.
GROUND_HAZARDS = {"pothole", "drain", "manhole", "road_block", "road_hump", "pavement_marking", "garbage"}
MOVING_OBJECTS  = {"person", "rider", "dog", "cow", "buffalo", "mule", "car",
                   "motorcycle", "autorickshaw", "bus", "truck", "bicycle",
                   "e-rickshaw", "rickshaw", "hand_cart"}


# ─────────────────────────────────────────────
# Distance urgency thresholds (in cm)
# ─────────────────────────────────────────────
URGENCY_THRESHOLDS = {
    "CRITICAL": 100,   # < 1 m  → immediate action
    "WARNING":  250,   # < 2.5 m
    "CAUTION":  500,   # < 5 m
}

def get_urgency(distance_cm):
    if distance_cm < URGENCY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif distance_cm < URGENCY_THRESHOLDS["WARNING"]:
        return "WARNING"
    elif distance_cm < URGENCY_THRESHOLDS["CAUTION"]:
        return "CAUTION"
    else:
        return "INFO"


# ─────────────────────────────────────────────
# Lateral zone from bounding box center
# ─────────────────────────────────────────────
def get_lateral_zone(bbox_cx, frame_w):
    """Returns: 'left', 'center', or 'right'"""
    third = frame_w / 3
    if bbox_cx < third:
        return "left"
    elif bbox_cx < 2 * third:
        return "center"
    else:
        return "right"


# ─────────────────────────────────────────────
# Action suggestion logic
# ─────────────────────────────────────────────
def get_action(label, position, urgency):
    """
    Returns a short action string based on:
    - where the object is (left/center/right)
    - what the object is (ground hazard vs obstacle)
    - urgency level
    """
    # Ground hazards: step to either side
    if label in GROUND_HAZARDS:
        if position == "center":
            return "stop and move aside"
        elif position == "left":
            return "move to your right"
        else:
            return "move to your left"

    # Traffic signals
    if label in ("light_red", "light_ped_stop"):
        return "stop"
    if label in ("light_green", "light_ped_walk"):
        return "you may proceed"
    if label == "light_yellow":
        return "slow down"

    # Moving objects and general obstacles
    if urgency == "CRITICAL":
        if position == "left":
            return "move right immediately"
        elif position == "right":
            return "move left immediately"
        else:
            return "stop immediately"
    elif urgency == "WARNING":
        if position == "left":
            return "move to your right"
        elif position == "right":
            return "move to your left"
        else:
            return "slow down and proceed carefully"
    else:
        if position == "left":
            return "keep right"
        elif position == "right":
            return "keep left"
        else:
            return "proceed with caution"


# ─────────────────────────────────────────────
# Core phrase builder
# ─────────────────────────────────────────────
def build_phrase(label, position, distance_cm, urgency):
    """
    Builds a natural language navigation instruction.

    Examples:
      "Pothole ahead, 80 centimetres. Stop and move aside."
      "Warning. Person on your left, 2 metres. Move to your right."
      "Auto-rickshaw on your right, 4 metres. Keep left."
    """
    name    = FRIENDLY_NAMES.get(label, label.replace("_", " "))
    action  = get_action(label, position, urgency)

    # Distance phrasing
    if distance_cm < 100:
        dist_str = f"{int(distance_cm)} centimetres"
    else:
        metres = distance_cm / 100
        dist_str = f"{metres:.1f} metres".replace(".0 ", " ")

    # Position phrasing
    if position == "center":
        pos_str = "ahead"
    else:
        pos_str = f"on your {position}"

    # Urgency prefix
    if urgency == "CRITICAL":
        prefix = "Danger! "
    elif urgency == "WARNING":
        prefix = "Warning. "
    else:
        prefix = ""

    phrase = f"{prefix}{name.capitalize()} {pos_str}, {dist_str}. {action.capitalize()}."
    return phrase


# ─────────────────────────────────────────────
# Phrase deduplication / cooldown tracker
# ─────────────────────────────────────────────
class AnnouncementCooldown:
    """
    Prevents the same phrase from being repeated too frequently.
    Each label+zone combo has its own cooldown timer.
    """
    def __init__(self, cooldown_sec=4.0):
        self.cooldown_sec = cooldown_sec
        self._last_spoken = {}   # key → timestamp

    def should_speak(self, label, position, urgency):
        # Critical alerts have a shorter cooldown (1 s)
        cd = 1.0 if urgency == "CRITICAL" else self.cooldown_sec
        key = f"{label}_{position}"
        now = time.time()
        if key not in self._last_spoken or (now - self._last_spoken[key]) >= cd:
            self._last_spoken[key] = now
            return True
        return False


# ─────────────────────────────────────────────
# TTS engine (runs in a background thread)
# ─────────────────────────────────────────────
class TTSEngine:
    """
    Wraps pyttsx3 in a background thread with a priority queue so that
    CRITICAL messages interrupt lower-priority ones.
    Priority values: 0 = highest (CRITICAL), 3 = lowest (INFO)
    """
    PRIORITY_MAP = {"CRITICAL": 0, "WARNING": 1, "CAUTION": 2, "INFO": 3}

    def __init__(self, rate=145, volume=1.0):
        self._q = queue.PriorityQueue()
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, phrase, urgency="CAUTION"):
        priority = self.PRIORITY_MAP.get(urgency, 3)
        self._q.put((priority, phrase))

    def _worker(self):
        while True:
            _, phrase = self._q.get()
            try:
                self._engine.say(phrase)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[TTS error] {e}")

    def wait_until_done(self):
        self._q.join()


# ─────────────────────────────────────────────
# Horizon check
# ─────────────────────────────────────────────
def get_horizon_y(pitch_rad, roll_rad=1.0):
    horizon_y = 230
    return horizon_y


# ─────────────────────────────────────────────
# Distance estimation (below-horizon only)
# ─────────────────────────────────────────────
def calculate_distance(pixel_x, pixel_y, roll_rad, pitch_rad, h_cm):
    dx = pixel_x - CX
    dy = pixel_y - CY

    y_corrected = dx * math.sin(roll_rad) + dy * math.cos(roll_rad)
    pixel_angle_offset = (y_corrected / IMAGE_H) * V_FOV_RAD

    total_angle = abs(pitch_rad) - pixel_angle_offset
    if total_angle <= 0:
        return None

    return h_cm / math.tan(total_angle)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_navigation_assistant(source, roll_rad, pitch_rad, h_cm=H_CM,
                              max_announcements=3, tts_rate=145):
    """
    Full pipeline: YOLO detection → distance estimation → phrase generation → TTS.

    Args:
        source            : image path, video path, or 0 for webcam
        roll_rad          : camera roll in radians
        pitch_rad         : camera pitch in radians (negative = tilted down)
        h_cm              : camera height from ground in cm
        max_announcements : max phrases to speak per frame (avoids overwhelming user)
        tts_rate          : speech rate (words per minute)
    """
    model   = YOLO("/Users/firaaskhan/Desktop/majr/yolo11s.pt")
    tts     = TTSEngine(rate=tts_rate)
    cooldown = AnnouncementCooldown(cooldown_sec=4.0)

    horizon_y = get_horizon_y(pitch_rad, roll_rad)
    print(f"Horizon line at y = {horizon_y:.1f}px  (image height = {IMAGE_H}px)")
    print("─" * 60)

    results_all = model.predict(source=source, conf=0.4, verbose=False)

    for frame_idx, result in enumerate(results_all):
        print(f"\n[Frame {frame_idx}]")

        # ── Collect valid detections for this frame ──
        candidates = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = box.conf[0].item()
            cls   = int(box.cls[0].item())
            label = result.names[cls]

            bx = (x1 + x2) / 2   # bottom-center x
            by = y2               # bottom-center y (ground contact)

            if by <= horizon_y:
                print(f"  SKIP  [{label:18s}] bottom_y={by:.0f} above horizon")
                continue

            distance_cm = calculate_distance(bx, by, roll_rad, pitch_rad, h_cm)
            if distance_cm is None:
                continue

            position = get_lateral_zone(bx, IMAGE_W)
            urgency  = get_urgency(distance_cm)
            priority = OBJECT_PRIORITY.get(label, 99)

            candidates.append({
                "label":       label,
                "conf":        conf,
                "distance_cm": distance_cm,
                "position":    position,
                "urgency":     urgency,
                "priority":    priority,
            })

            print(f"  VALID [{label:18s}] conf={conf:.2f} | "
                  f"pos={position:6s} | dist={distance_cm:6.1f} cm | "
                  f"urgency={urgency}")

        if not candidates:
            print("  (no objects detected below horizon)")
            continue

        # ── Sort: first by object priority, then by distance (closer first) ──
        candidates.sort(key=lambda d: (d["priority"], d["distance_cm"]))

        # ── Generate and speak top N phrases ──
        spoken_count = 0
        print()
        for det in candidates:
            if spoken_count >= max_announcements:
                break

            label    = det["label"]
            position = det["position"]
            urgency  = det["urgency"]

            if not cooldown.should_speak(label, position, urgency):
                continue

            phrase = build_phrase(label, position, det["distance_cm"], urgency)
            print(f"  🔊 [{urgency:8s}] {phrase}")

            tts.speak(phrase, urgency=urgency)
            spoken_count += 1

    # Wait for all speech to finish before exiting
    print("\n" + "─" * 60)
    print("Processing complete. Waiting for speech to finish...")
    tts.wait_until_done()


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_navigation_assistant(
        source="/Users/firaaskhan/Desktop/majr/try.png",   # image path, video path, or 0 for webcam
        roll_rad=0.0,
        pitch_rad=0.0,
        h_cm=145,
        max_announcements=3,        # speak at most 3 warnings per frame
        tts_rate=145,               # slightly slower for clarity
    )


# ─────────────────────────────────────────────
# Example phrase outputs (for reference)
# ─────────────────────────────────────────────
"""
CRITICAL examples:
  "Danger! Pothole ahead, 65 centimetres. Stop and move aside."
  "Danger! Person on your left, 80 centimetres. Move right immediately."
  "Danger! Dog on your right, 90 centimetres. Move left immediately."

WARNING examples:
  "Warning. Open drain on your left, 1.8 metres. Move to your right."
  "Warning. Auto-rickshaw ahead, 2 metres. Slow down and proceed carefully."
  "Warning. Cow on your right, 2.2 metres. Move to your left."

CAUTION examples:
  "Speed bump ahead, 3.5 metres. Slow down."
  "Person on your left, 4 metres. Keep right."
  "Motorcycle on your right, 4.5 metres. Keep left."

INFO / traffic signals:
  "Red signal ahead, 6 metres. Stop."
  "Green signal ahead, 5 metres. You may proceed."
"""