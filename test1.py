import math
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Camera parameters
# ─────────────────────────────────────────────
IMAGE_W, IMAGE_H = 512, 512
CX, CY = IMAGE_W // 2, IMAGE_H // 2
V_FOV_RAD = math.radians(170)
H_CM = 145  # camera height in cm


# ─────────────────────────────────────────────
# Horizon check — where is the horizon pixel?
# ─────────────────────────────────────────────
def get_horizon_y(pitch_rad, roll_rad=1.0):
    """
    Returns the Y pixel coordinate of the horizon line.
    Any pixel with by > horizon_y is below the horizon (ground-visible).
    """
    # Angle from center to horizon = pitch magnitude
    # Reverse of pixel_angle_offset formula: y = (angle / V_FOV_RAD) * IMAGE_H
    # horizon_offset = (abs(pitch_rad) / V_FOV_RAD) * IMAGE_H
    # horizon_y = CY + horizon_offset  # below center since camera tilts down
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
        return None  # above horizon — skip

    return h_cm / math.tan(total_angle)


# ─────────────────────────────────────────────
# Main detection + distance
# ─────────────────────────────────────────────
def run_distance_estimation(source, roll_rad, pitch_rad, h_cm=H_CM):
    model = YOLO("D:\\Projects\\Major Project\\yolo11s.pt")

    # Horizon Y for this camera orientation
    horizon_y = get_horizon_y(pitch_rad, roll_rad)
    print(f"Horizon line at y = {horizon_y:.1f}px  (image height = {IMAGE_H}px)")

    results_all = model.predict(source=source, conf=0.4, verbose=False)

    for result in results_all:
        print(f"\n{'─'*55}")

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = box.conf[0].item()
            cls   = int(box.cls[0].item())
            label = result.names[cls]

            # ── Bottom-center: contact point with the ground ──
            bx = (x1 + x2) / 2
            by = y2  # bottom edge of bbox

            # ── Filter: only process if bottom edge is below the horizon ──
            if by <= horizon_y:
                print(f"  SKIP  [{label:12s}] conf={conf:.2f} | "
                      f"bottom y={by:.0f} is above/at horizon ({horizon_y:.0f})")
                continue

            # ── Estimate distance ──
            distance_cm = calculate_distance(bx, by, roll_rad, pitch_rad, h_cm)

            if distance_cm is None:
                # Shouldn't happen after the horizon check above, but guard anyway
                print(f"  SKIP  [{label:12s}] conf={conf:.2f} | "
                      f"total_angle <= 0 (degenerate case)")
                continue

            print(f"  VALID [{label:12s}] conf={conf:.2f} | "
                  f"bottom_center=({bx:.0f}, {by:.0f}) | "
                  f"distance={distance_cm:.1f} cm ({distance_cm/100:.2f} m)")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_distance_estimation(
        source="Images/2047.jpg",   # image, video path, or 0 for webcam
        roll_rad=0,
        pitch_rad=0,
        h_cm=145
    )

'''
## Dry run trace

With `pitch=-0.49976 rad`, `IMAGE_H=720`, `V_FOV_RAD=0.7505`:
```
horizon_y = 360 + (0.49976 / 0.7505) * 720
          = 360 + 0.6659 * 720
          = 360 + 479.4
          = 839.4   ← beyond the bottom of the frame (720px)
```

This means with a ~28.6° downward tilt, the **entire frame is below the horizon** — every detected object passes the filter and gets a distance estimate. This makes sense for a dashcam or helmet-mounted camera pointing strongly downward.

If you reduce the tilt to something shallow like `pitch=-0.15 rad`:
```
horizon_y = 360 + (0.15 / 0.7505) * 720 = 360 + 144 = 504px
'''
