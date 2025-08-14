# streamlit_app.py
# ASCII-only, concise comments. Streamlit + streamlit-webrtc + MediaPipe.
# Shows 21 achromatic circles with descending Weber contrast on a gray pedestal.
# Live webcam "keep gaze fixed" banner using face angle/center drift.

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import csv, io, time, random, threading

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import cv2
import mediapipe as mp


# ---------------------- Contrast utils (linear <-> sRGB) ----------------------

def linear_to_srgb(v):
    v = max(0.0, min(1.0, float(v)))
    if v <= 0.0031308:
        return 12.92 * v
    else:
        return 1.055 * (v ** (1.0 / 2.4)) - 0.055

def gray_rgb_from_linear(L):
    s = linear_to_srgb(L)
    x = int(round(s * 255))
    x = max(0, min(255, x))
    return (x, x, x)

def compute_L_target(L_bg, C, brighter=True):
    if brighter:
        L_t = L_bg * (1.0 + C)
    else:
        L_t = L_bg * (1.0 - C)
    return max(0.0, min(1.0, L_t))


# ---------------------- Gaze processor (MediaPipe FaceDetection) ---------------

class FaceProcessor(VideoProcessorBase):
    """
    Processes webcam frames to estimate drift vs a baseline center/roll.
    Shows an overlay when drifting. Baseline set by a timed calibration.
    """
    def __init__(self):
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )
        self.frame_w = 1
        self.frame_h = 1

        # Smoothing
        self.alpha = 0.55
        self.smooth_center = None  # np.array([cx, cy])
        self.smooth_angle = None   # np.array([roll, yaw])

        # Baseline (cx, cy, roll)
        self.baseline = None

        # Drift thresholds
        self.max_center_shift_pct = 0.015
        self.max_angle_deg = 2.0
        self.dwell_ms_required = 80
        self._drift_ms = 0.0
        self._last_t = time.time()

        # Calibration control
        self._cal_lock = threading.Lock()
        self._cal_samples = []
        self._cal_angles = []
        self._cal_until = 0.0
        self.calibrating = False
        self.cal_status = "Press Calibrate to set baseline."

        # Public status for the UI
        self.last_state = "unknown"
        self.last_metrics = {"center_shift_pct": None, "angle_diff_deg": None, "dwell_ms": 0}

    def start_calibration(self, duration_sec=1.2):
        with self._cal_lock:
            self._cal_samples = []
            self._cal_angles = []
            self._cal_until = time.time() + float(duration_sec)
            self.calibrating = True
            self.cal_status = "Calibrating..."

    def _finish_calibration_if_ready(self):
        with self._cal_lock:
            if not self.calibrating:
                return
            if time.time() < self._cal_until:
                return
            if not self._cal_samples:
                self.cal_status = "Calibration failed: no face detected."
                self.calibrating = False
                return
            base_center = np.median(np.stack(self._cal_samples, axis=0), axis=0)
            base_angle = 0.0
            if self._cal_angles:
                base_angle = float(np.median(np.stack(self._cal_angles, axis=0), axis=0)[0])
            self.baseline = (float(base_center[0]), float(base_center[1]), base_angle)
            self._drift_ms = 0.0
            self._last_t = time.time()
            self.calibrating = False
            self.cal_status = "Calibration OK. Keep gaze fixed."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        self.frame_h, self.frame_w = h, w

        # Face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face.process(rgb)

        cx = cy = None
        yaw = roll = None

        if res.detections:
            det = res.detections[0]
            bbox = det.location_data.relative_bounding_box
            cx = (bbox.xmin + bbox.width / 2.0) * self.frame_w
            cy = (bbox.ymin + bbox.height / 2.0) * self.frame_h

            kps = det.location_data.relative_keypoints
            if len(kps) >= 2:
                rx = kps[0].x * self.frame_w
                ry = kps[0].y * self.frame_h
                lx = kps[1].x * self.frame_w
                ly = kps[1].y * self.frame_h
                roll = float(np.degrees(np.arctan2((ly - ry), (lx - rx))))
                ex_mid_x = (lx + rx) / 2.0
                eye_dx = max(1.0, abs(lx - rx))
                yaw = float(np.degrees(np.arctan2((cx - ex_mid_x), eye_dx)))

        # Exponential smoothing
        if cx is not None:
            if self.smooth_center is None:
                self.smooth_center = np.array([cx, cy], dtype=float)
            else:
                self.smooth_center = self.alpha * np.array([cx, cy]) + (1 - self.alpha) * self.smooth_center

        if roll is not None:
            if self.smooth_angle is None:
                self.smooth_angle = np.array([roll, yaw if yaw is not None else 0.0], dtype=float)
            else:
                cur = np.array([roll, yaw if yaw is not None else self.smooth_angle[1]], dtype=float)
                self.smooth_angle = self.alpha * cur + (1 - self.alpha) * self.smooth_angle

        # Collect calibration samples if active
        if self.calibrating and self.smooth_center is not None:
            with self._cal_lock:
                self._cal_samples.append(self.smooth_center.copy())
                if self.smooth_angle is not None:
                    self._cal_angles.append(self.smooth_angle.copy())
        self._finish_calibration_if_ready()

        # Drift state
        now = time.time()
        if self.baseline is None or self.smooth_center is None:
            state = "unknown"
            metrics = {"center_shift_pct": None, "angle_diff_deg": None, "dwell_ms": 0}
        else:
            base_cx, base_cy, base_angle = self.baseline
            dx = (self.smooth_center[0] - base_cx) / max(1.0, self.frame_w)
            dy = (self.smooth_center[1] - base_cy) / max(1.0, self.frame_h)
            center_shift_pct = float(np.hypot(dx, dy))
            angle_now = float(self.smooth_angle[0]) if self.smooth_angle is not None else base_angle
            angle_diff = abs(angle_now - base_angle)

            drifting = (center_shift_pct > self.max_center_shift_pct) or (angle_diff > self.max_angle_deg)

            dt_ms = (now - self._last_t) * 1000.0
            self._last_t = now
            if drifting:
                self._drift_ms += dt_ms
            else:
                self._drift_ms = max(0.0, self._drift_ms - dt_ms * 0.5)

            if self._drift_ms >= self.dwell_ms_required:
                state = "drifting"
            else:
                state = "stable"
            metrics = {
                "center_shift_pct": center_shift_pct,
                "angle_diff_deg": angle_diff,
                "dwell_ms": int(self._drift_ms),
            }

        # Save public status
        self.last_state = state
        self.last_metrics = metrics

        # Overlay text
        if self.calibrating:
            txt = "Calibrating..."
            color = (200, 200, 50)
        elif state == "drifting":
            txt = "Keep gaze fixed on the center"
            color = (30, 30, 230)  # red-ish in BGR
        elif state == "unknown":
            txt = "Face not detected"
            color = (80, 180, 255)
        else:
            txt = "Gaze OK"
            color = (80, 200, 80)

        cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        # Small center dot overlay
        cv2.circle(img, (self.frame_w // 2, self.frame_h // 2), 6, (0, 255, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------- Streamlit UI and stimulus loop ------------------------

st.set_page_config(page_title="Contrast Staircase + Gaze", layout="wide")
st.title("Contrast Staircase (21 circles) with Webcam Gaze Banner")

# Left: controls and webcam; Right: stimulus and logs
left, right = st.columns([1, 1.4])

# Session state
ss = st.session_state
if "running" not in ss:
    ss.running = False
if "circle_count" not in ss:
    ss.circle_count = 0
if "contrast" not in ss:
    ss.contrast = 0.80
if "rows" not in ss:
    ss.rows = []
if "rng" not in ss:
    ss.rng = random.Random(42)  # deterministic locations
if "last_run_done" not in ss:
    ss.last_run_done = False

# Controls
with left:
    st.subheader("Webcam gaze")
    webrtc_ctx = webrtc_streamer(
        key="gaze",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    cal_btn = st.button("Calibrate gaze (1.2 s)")
    if webrtc_ctx and webrtc_ctx.video_processor:
        vp = webrtc_ctx.video_processor
        if cal_btn:
            vp.start_calibration(1.2)
        st.caption(vp.cal_status)
        st.write("Gaze state:", vp.last_state)
        st.write(vp.last_metrics)

    st.subheader("Stimulus settings")
    L_bg = st.slider("Background luminance (linear)", 0.1, 0.9, 0.4, 0.05)
    min_contrast = 0.005
    max_circles = 21
    contrast_step = (0.80 - min_contrast) / 20.0  # reach 0.5% at the 21st

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.button("Start test", type="primary", disabled=ss.running)
    with c2:
        reset = st.button("Reset")
    with c3:
        pace_s = st.number_input("Seconds per circle", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

    if reset:
        ss.running = False
        ss.circle_count = 0
        ss.contrast = 0.80
        ss.rows = []
        ss.last_run_done = False
        st.info("Reset done.")

with right:
    st.subheader("Stimulus")
    canvas = st.empty()
    status = st.empty()
    resp_cols = st.columns(2)
    dl_slot = st.empty()

# Start the run
if start:
    ss.running = True
    ss.circle_count = 0
    ss.contrast = 0.80
    ss.rows = []
    ss.last_run_done = False
    status.info("Running...")

# Stimulus loop (blocks while drawing; video runs in parallel thread)
if ss.running:
    # Fixed canvas size for consistency on all screens
    W, H = 900, 600
    size = 90
    bg_rgb = gray_rgb_from_linear(L_bg)

    while ss.running and ss.circle_count < max_circles:
        # Compute Weber target
        C = max(min_contrast, ss.contrast)
        L_target = compute_L_target(L_bg, C, brighter=True)
        tgt_rgb = gray_rgb_from_linear(L_target)

        # Make background and draw one circle
        img = Image.new("RGB", (W, H), color=bg_rgb)
        draw = ImageDraw.Draw(img)

        def near_edge(x, y, margin=50):
            return x < margin or x > W - margin - size or y < margin or y > H - margin - size

        # Pick location (95% near edge)
        while True:
            x1 = ss.rng.randint(0, W - size)
            y1 = ss.rng.randint(0, H - size)
            if ss.rng.random() < 0.95:
                if near_edge(x1, y1):
                    break
            else:
                break

        x2, y2 = x1 + size, y1 + size
        draw.ellipse([x1, y1, x2, y2], fill=tgt_rgb)

        # Yellow fixation dot at center
        fx, fy, r = W // 2, H // 2, 10
        draw.ellipse([fx - r, fy - r, fx + r, fy + r], fill=(255, 255, 0))

        # Show image
        canvas.image(img, use_column_width=True)

        # Log row (similar to desktop CSV)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        dx, dy = mx - (W // 2), (H // 2) - my
        angle = np.degrees(np.arctan2(dy, dx))
        angle = float(angle if angle >= 0 else angle + 360)
        color_hex = "#{:02x}{:02x}{:02x}".format(*tgt_rgb)
        ss.rows.append(["circle", color_hex, mx, my, angle, float(C), float(L_target), ""])

        # Step and count
        ss.contrast = max(min_contrast, ss.contrast - contrast_step)
        ss.circle_count += 1

        # Pace
        time.sleep(float(pace_s))

    # Done
    ss.running = False
    ss.last_run_done = True
    status.success("Done (21 circles).")

# Response buttons (optional)
if not ss.running:
    with resp_cols[0]:
        if st.button("Yes (detected)"):
            if ss.rows:
                ss.rows[-1][-1] = "Yes"
    with resp_cols[1]:
        if st.button("No (not detected)"):
            if ss.rows:
                ss.rows[-1][-1] = "No"

# Download CSV after a run
if ss.last_run_done and ss.rows:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Shape Type", "Color Hex", "Center_x", "Center_y", "Angle_deg",
        "Weber_Contrast_C", "L_target_linear", "User Response"
    ])
    writer.writerows(ss.rows)
    dl_slot.download_button("Download CSV", buf.getvalue(),
                            file_name="contrast_shape_log.csv", mime="text/csv")
