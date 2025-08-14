# -*- coding: utf-8 -*-
"""
Random Circle Popper with contrast-based grayscale staircase and webcam-based
"keep gaze fixed" warning. Achromatic stimuli on a gray pedestal background.
ASCII-only script to avoid encoding issues on Windows.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "3")

import tkinter as tk
from tkinter import simpledialog
import random
import math
import csv
import sys
import time
import threading
import platform

import numpy as np
import cv2
import mediapipe as mp


# ------------------------------ Utility: platform ------------------------------

def is_wsl():
    """Detect if running under Windows Subsystem for Linux (WSL)."""
    try:
        return ("microsoft" in platform.release().lower()) or ("WSL_DISTRO_NAME" in os.environ)
    except Exception:
        return False


# ------------------------------ Utility: camera opener ------------------------------

class CameraOpener:
    """
    Tries multiple camera indices and backends to find a working webcam.
    Stores which backend/index succeeded for reference.
    """
    def __init__(self):
        self.backend_used = None
        self.index_used = None

    def open(self, preferred_index=0, max_indices=5):
        sysplat = sys.platform
        if sysplat.startswith("win"):
            candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sysplat == "darwin":
            candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]

        for idx in range(preferred_index, preferred_index + max_indices):
            for backend in candidates:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self.backend_used = backend
                        self.index_used = idx
                        return cap
                try:
                    cap.release()
                except Exception:
                    pass
        return None


# ------------------------------ Gaze monitor (MediaPipe FaceDetection) ------------------------------

class GazeNoHardware:
    """
    Webcam-based head/eye-line drift monitor using MediaPipe FaceDetection.
    Not clinical eye tracking. Suitable for showing a "keep gaze fixed" banner.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.opener = CameraOpener()

        # model_selection=0 for typical webcam distance
        self.face = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )

        self.running = False
        self.lock = threading.Lock()
        self.latest = None      # (cx, cy, yaw, roll, t)
        self.baseline = None    # (cx, cy, roll)
        self.smooth_center = None
        self.smooth_angle = None
        self.alpha = 0.55

        self.frame_w = 1
        self.frame_h = 1

        # Tuning thresholds (conservative for "keep gaze" banner)
        self.max_center_shift_pct = 0.015
        self.max_angle_deg = 2.0
        self.dwell_ms_required = 80
        self._drift_ms = 0
        self._last_t = time.time()

        self.fail_reason = None

    def start(self):
        if self.running:
            return
        if is_wsl():
            self.fail_reason = "Running inside WSL. Webcam devices are not available. Use Windows Python."
            raise RuntimeError(self.fail_reason)

        cap = self.opener.open(preferred_index=self.camera_index, max_indices=5)
        if cap is None:
            self.fail_reason = "No webcam found or it is busy. Close Zoom/Teams and run from native Windows."
            raise RuntimeError("Webcam open failed")

        self.cap = cap
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            self.frame_h, self.frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face.process(rgb)

            cx = cy = None
            yaw = roll = None
            t = time.time()

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
                    # Roll: angle of line between eyes
                    roll = float(np.degrees(np.arctan2((ly - ry), (lx - rx))))
                    # Yaw: horizontal offset normalized by eye distance
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

            with self.lock:
                self.latest = (cx, cy, yaw, roll, t)

            time.sleep(0.005)

    def calibrate(self, duration_sec=1.0):
        """
        Sample for duration_sec while the user looks at the center dot.
        Set baseline center and baseline eye-roll angle.
        """
        samples = []
        angles = []
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            with self.lock:
                cxcy = self.smooth_center.copy() if self.smooth_center is not None else None
                ang = self.smooth_angle.copy() if self.smooth_angle is not None else None
            if cxcy is not None:
                samples.append(cxcy)
            if ang is not None:
                angles.append(ang)
            time.sleep(0.01)

        if not samples:
            raise RuntimeError("Calibration failed: no face detected")

        base_center = np.median(np.stack(samples, axis=0), axis=0)
        base_angle = float(np.median(np.stack(angles, axis=0), axis=0)[0]) if angles else 0.0

        self.baseline = (float(base_center[0]), float(base_center[1]), base_angle)
        self._drift_ms = 0
        self._last_t = time.time()
        return self.baseline

    def drift_state(self):
        """
        Returns tuple (state, metrics)
        state in {"stable", "drifting", "unknown"}
        metrics: dict with keys center_shift_pct, angle_diff_deg, dwell_ms
        """
        now = time.time()
        with self.lock:
            cxcy = self.smooth_center.copy() if self.smooth_center is not None else None
            ang = self.smooth_angle.copy() if self.smooth_angle is not None else None

        if self.baseline is None or cxcy is None:
            return "unknown", {"center_shift_pct": None, "angle_diff_deg": None, "dwell_ms": 0}

        base_cx, base_cy, base_angle = self.baseline
        dx = (cxcy[0] - base_cx) / max(1.0, self.frame_w)
        dy = (cxcy[1] - base_cy) / max(1.0, self.frame_h)
        center_shift_pct = float(np.hypot(dx, dy))
        angle_now = float(ang[0]) if ang is not None else base_angle
        angle_diff = abs(angle_now - base_angle)

        drifting = (center_shift_pct > self.max_center_shift_pct) or (angle_diff > self.max_angle_deg)

        dt_ms = (now - self._last_t) * 1000.0
        self._last_t = now
        if drifting:
            self._drift_ms += dt_ms
        else:
            self._drift_ms = max(0, self._drift_ms - dt_ms * 0.5)

        if self._drift_ms >= self.dwell_ms_required:
            return "drifting", {
                "center_shift_pct": center_shift_pct,
                "angle_diff_deg": angle_diff,
                "dwell_ms": int(self._drift_ms),
            }
        else:
            return "stable", {
                "center_shift_pct": center_shift_pct,
                "angle_diff_deg": angle_diff,
                "dwell_ms": int(self._drift_ms),
            }


# ------------------------------ Utility: color space (linear <-> sRGB) ------------------------------

def linear_to_srgb(v):
    """
    Convert scalar luminance in linear light [0,1] to sRGB-encoded value [0,1].
    Uses standard sRGB electro-optical transfer function.
    """
    v = max(0.0, min(1.0, float(v)))
    if v <= 0.0031308:
        return 12.92 * v
    else:
        return 1.055 * (v ** (1.0 / 2.4)) - 0.055


def srgb_to_linear(vs):
    """
    Convert sRGB-encoded value [0,1] to linear light [0,1].
    """
    vs = max(0.0, min(1.0, float(vs)))
    if vs <= 0.04045:
        return vs / 12.92
    else:
        return ((vs + 0.055) / 1.055) ** 2.4


def gray_hex_from_linear(L):
    """
    Given linear luminance L in [0,1], return a Tkinter hex like "#aabbcc" with R=G=B.
    """
    s = linear_to_srgb(L)
    x = int(round(s * 255))
    x = max(0, min(255, x))
    return f"#{x:02x}{x:02x}{x:02x}"


def hex_from_srgb_triplet(r, g, b):
    """
    Given sRGB-encoded triplet in [0,1], return hex string.
    """
    r8 = max(0, min(255, int(round(r * 255))))
    g8 = max(0, min(255, int(round(g * 255))))
    b8 = max(0, min(255, int(round(b * 255))))
    return f"#{r8:02x}{g8:02x}{b8:02x}"


# ------------------------------ Main app with contrast-based staircase ------------------------------

class CircleApp:
    """
    Displays achromatic circles with luminance defined by Weber contrast relative to a gray background.
    Contrast follows a descending staircase across trials (monotonic decrease).
    Automatically closes after the 21st circle appears.
    """
    def __init__(self, root, log_filename):
        self.root = root
        self.log_filename = log_filename
        self.pending_jobs = []
        self.is_closing = False

        # Current shape queued for response logging:
        # (type, color_hex, cx, cy, angle_deg, contrast, L_target_linear)
        self.current_shape_data = None

        # ---------------- Contrast staircase parameters ----------------
        self.L_bg = 0.4                            # background luminance (linear)
        self.contrast = 0.80                       # initial Weber contrast
        self.contrast_step = (0.80 - 0.005) / 20   # reach 0.5% at the 21st circle
        self.min_contrast = 0.005                  # floor (0.5%)
        self.present_brighter = True               # brighter-than-background

        # ---------------- Trial counting (auto-close at 21) ------------
        self.circle_count = 0
        self.max_circles = 21

        self.root.title("Random Circle Popper (Contrast Staircase)")
        self.setup_window()
        self.setup_canvas()
        self.setup_logging()
        self.setup_keybindings()

        # Webcam gaze monitor
        self.gaze = None
        self.setup_gaze_monitor()

        self.root.after(100, self.initial_draw)

    # ---------------- Window and canvas ----------------

    def setup_window(self):
        if sys.platform == "darwin":
            self.root.attributes("-fullscreen", True)
        elif sys.platform.startswith("win"):
            self.root.state("zoomed")
        else:
            try:
                self.root.attributes("-zoomed", True)
            except tk.TclError:
                pass
        self.root.update_idletasks()
        self.root.minsize(600, 400)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_canvas(self):
        bg_hex = gray_hex_from_linear(self.L_bg)  # gray pedestal
        self.canvas = tk.Canvas(self.root, bg=bg_hex, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        self.draw_permanent_circle()

    def draw_permanent_circle(self):
        """Draw a small yellow fixation patch at the screen center."""
        self.canvas.delete("permanent")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w > 100 and h > 100:
            size = 20
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size
            yellow_hex = hex_from_srgb_triplet(1.0, 1.0, 0.0)
            self.canvas.create_oval(x1, y1, x2, y2, fill=yellow_hex, outline="", tags="permanent")

    # ---------------- Logging ----------------

    def setup_logging(self):
        self.csv_file = open(self.log_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Shape Type", "Color Hex", "Center_x", "Center_y", "Angle_deg",
            "Weber_Contrast_C", "L_target_linear", "User Response"
        ])

    # ---------------- Key bindings ----------------

    def setup_keybindings(self):
        self.root.bind("<Up>",   lambda e: self.log_and_clear("Yes"))
        self.root.bind("<Down>", lambda e: self.log_and_clear("No"))
        self.root.bind("c", lambda e: self.calibrate_gaze())

    def log_and_clear(self, response):
        if self.current_shape_data is not None:
            self.csv_writer.writerow([*self.current_shape_data, response])
            self.csv_file.flush()
            self.current_shape_data = None

    # ---------------- Trial scheduling and drawing ----------------

    def initial_draw(self):
        self.draw_permanent_circle()
        if self.gaze is not None and getattr(self.gaze, "running", False):
            self.canvas.delete("msg")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 30,
                text="Keep eyes on the center dot (press 'c' to recalibrate).",
                fill="#aaaaaa", font=("Arial", 14), tags="msg"
            )
        self.root.after(1000, self.schedule_first_draw)

    def schedule_first_draw(self):
        job = self.root.after(0, self.draw_shape)
        self.pending_jobs.append(job)

    def calculate_angle(self, x, y, cx, cy):
        dx = x - cx
        dy = cy - y
        angle = math.degrees(math.atan2(dy, dx))
        return angle if angle >= 0 else angle + 360

    def is_near_edge(self, x, y, w, h, margin=50):
        return x < margin or x > w - margin or y < margin or y > h - margin

    def _compute_target_luminance(self):
        """
        Compute target luminance in linear light from current contrast and background.
        Uses Weber contrast. Clips to [0,1] to avoid out-of-gamut issues.
        """
        C = max(self.min_contrast, self.contrast)
        if self.present_brighter:
            L_target = self.L_bg * (1.0 + C)
        else:
            L_target = self.L_bg * (1.0 - C)
        L_target = max(0.0, min(1.0, L_target))
        return C, L_target

    def draw_shape(self):
        if self.is_closing:
            return

        # Log missed response if previous trial was not answered
        if self.current_shape_data:
            self.csv_writer.writerow([*self.current_shape_data, "NR"])
            self.current_shape_data = None

        self.canvas.delete("temporary")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        if w > 100 and h > 100:
            cx = w // 2
            cy = h // 2
            size = 90

            # Compute contrast-defined luminance and map to sRGB gray
            C, L_target = self._compute_target_luminance()
            hex_color = gray_hex_from_linear(L_target)

            # Pick location (biased near edges 95% of the time)
            if random.random() < 0.95:
                while True:
                    x1 = random.randint(0, w - size)
                    y1 = random.randint(0, h - size)
                    if self.is_near_edge(x1, y1, w, h):
                        break
            else:
                x1 = random.randint(0, w - size)
                y1 = random.randint(0, h - size)

            x2, y2 = x1 + size, y1 + size
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            angle = self.calculate_angle(mx, my, cx, cy)

            # Draw circle and record trial
            self.canvas.create_oval(x1, y1, x2, y2, fill=hex_color, outline="", tags="temporary")
            self.current_shape_data = ("circle", hex_color, mx, my, angle, C, L_target)

            # Descending staircase
            self.contrast = max(self.min_contrast, self.contrast - self.contrast_step)

            # ---------- Count and auto-close after 21st appearance ----------
            self.circle_count += 1
            if self.circle_count >= self.max_circles:
                # Allow the 21st circle to render, then exit cleanly
                self.root.after(100, self.on_closing)
                return

        # Schedule next trial (2 seconds)
        if not self.is_closing:
            job = self.root.after(2000, self.draw_shape)
            self.pending_jobs.append(job)

    # ---------------- Gaze monitor hooks ----------------

    def setup_gaze_monitor(self):
        self.gaze = GazeNoHardware()
        try:
            self.gaze.start()
        except Exception as e:
            msg = getattr(self.gaze, "fail_reason", str(e))
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 30,
                text=f"Gaze monitor unavailable: {msg}",
                fill="#ff8080", font=("Arial", 14), tags="msg"
            )
            self.gaze = None
            return
        self.root.after(800, self.calibrate_gaze)

    def calibrate_gaze(self):
        if self.gaze is None or not self.gaze.running:
            return
        self.draw_permanent_circle()
        self.canvas.delete("msg")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2, 30,
            text="Calibrating... Keep eyes on the center dot",
            fill="#dddddd", font=("Arial", 16), tags="msg"
        )
        self.root.update_idletasks()

        def _do_cal():
            try:
                self.gaze.calibrate(duration_sec=1.2)
                msg = "Calibration OK. Keep eyes on the center (press 'c' to recalibrate)."
                color = "#a0e0a0"
            except Exception as e:
                msg = f"Calibration failed: {e}"
                color = "#ff8080"
            self.root.after(0, lambda: self._finish_calibration(msg, color))

        threading.Thread(target=_do_cal, daemon=True).start()

    def _finish_calibration(self, msg, color):
        self.canvas.delete("msg")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2, 30,
            text=msg, fill=color, font=("Arial", 14), tags="msg"
        )
        self.root.after(33, self.poll_gaze_state)

    def poll_gaze_state(self):
        if self.is_closing or self.gaze is None:
            return
        state, metrics = self.gaze.drift_state()
        self.canvas.delete("gaze_warn")

        if state == "drifting":
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 60,
                text="Keep gaze fixed on the center",
                fill="#ff4040", font=("Arial", 18, "bold"), tags="gaze_warn"
            )
        elif state == "unknown":
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 60,
                text="Face not detected (check lighting and camera).",
                fill="#ffaa66", font=("Arial", 12), tags="gaze_warn"
            )

        self.root.after(33, self.poll_gaze_state)

    # ---------------- Cleanup ----------------

    def on_closing(self):
        self.is_closing = True
        for job in self.pending_jobs:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        if self.current_shape_data:
            self.csv_writer.writerow([*self.current_shape_data, "NR"])
        try:
            self.csv_file.close()
        except Exception:
            pass
        try:
            if getattr(self, "gaze", None):
                self.gaze.stop()
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass


# ------------------------------ App bootstrap ------------------------------

def get_filename():
    temp = tk.Tk()
    temp.withdraw()
    name = simpledialog.askstring("Input", "Please enter the log filename:")
    temp.destroy()
    return f"{name or 'contrast_shape_log'}.csv"


if __name__ == "__main__":
    log_filename = get_filename()
    root = tk.Tk()
    app = CircleApp(root, log_filename)
    root.mainloop()
