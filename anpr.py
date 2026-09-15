#!/usr/bin/env python3
"""
==============================================================================
 Headless ANPR (Automatic Number Plate Recognition) for Raspberry Pi 4B
 Camera:  Raspberry Pi Camera Module V1.3 (OV5647) via Picamera2/libcamera
 Output:  16x2 (or 20x4) character LCD over I2C, showing AUTHORIZED / DENIED
 Auth DB: Local CSV file of authorized plates (no server / no MQTT needed)
==============================================================================

WHAT CHANGED FROM THE ORIGINAL PC/GUI VERSION
------------------------------------------------------------------------------
 - Removed: customtkinter GUI, PIL/Tkinter video rendering, multi-camera
   Excel-driven RTSP grid, OpenCV CUDA / GStreamer RTSP handling.
 - Removed: MQTT (paho-mqtt) publish/subscribe for plate authorization and
   parking-capacity dashboard data.
 - Added:   Picamera2 capture loop for the local Pi Camera (CSI ribbon).
 - Added:   Local CSV-based authorization lookup (authorized_plates.csv).
 - Added:   I2C character LCD output (RPLCD) showing the plate and an
   AUTHORIZED / DENIED verdict.
 - Kept:    The YOLO (ncnn) plate-detector + EasyOCR reading pipeline and the
   `filter_reading()` plate-normalization logic, unchanged, since that part
   is hardware-independent.

ONE-TIME RASPBERRY PI SETUP
------------------------------------------------------------------------------
1. Enable the camera and I2C interfaces:
       sudo raspi-config
       -> Interface Options -> Camera -> Enable
       -> Interface Options -> I2C    -> Enable
   Reboot afterwards.

2. Find your LCD's I2C address (LCD must be wired to SDA/SCL, 5V, GND):
       sudo i2cdetect -y 1
   Note the two-hex-digit address shown (commonly 0x27 or 0x3f) and set
   LCD_I2C_ADDRESS below to match.

3. System packages:
       sudo apt update
       sudo apt install -y python3-picamera2 --no-install-recommends \
                            python3-pip python3-opencv i2c-tools

4. Python packages (a virtualenv with --system-site-packages is recommended
   so it can see the apt-installed picamera2):
       python3 -m venv --system-site-packages venv
       source venv/bin/activate
       pip install ultralytics easyocr RPLCD smbus2

   NOTE: EasyOCR is PyTorch-based and is heavy for a Pi 4B; expect roughly
   1-3 seconds per OCR call. This is a CPU-only board, so keep DET_THRESH
   and PROCESS_EVERY_N_FRAMES tuned to your needs (see CONFIG below).

5. Place your exported NCNN plate-detector model folder (e.g.
   "best_ncnn_model") in the same directory as this script, matching
   MODEL_PATH below.

6. Create/edit authorized_plates.csv (auto-created with a sample row on
   first run if missing) with one plate per line, in the SAME format that
   filter_reading() produces, e.g.:
       plate
       WP-1234
       CAB-5678

RUN
------------------------------------------------------------------------------
       python3 anpr_headless.py
   Stop with Ctrl+C (the LCD is cleared on exit).
==============================================================================
"""

import os
import re
import csv
import time
import signal
import sys
from datetime import datetime

import cv2
import numpy as np

from picamera2 import Picamera2
from ultralytics import YOLO
import easyocr
from RPLCD.i2c import CharLCD

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Site identity (kept for logging purposes, no server involved) ---
PARKID = "PARK001"
GATE = "1"
DIRECTION = "IN"

# --- Plate detector model (NCNN export, CPU-friendly on ARM) ---
MODEL_PATH = "best_ncnn_model"

# --- Camera ---
CAMERA_RESOLUTION = (640, 480)   # lower = faster on Pi 4B
PROCESS_EVERY_N_FRAMES = 1       # raise to 2-3 to reduce CPU load further

# --- Detection / OCR thresholds (same as original) ---
DET_THRESH = 0.7
OCR_THRESH = 0.98

# --- Authorization database ---
AUTHORIZED_PLATES_CSV = "authorized_plates.csv"

# --- Detection logging / saved crops (optional but kept from original) ---
LOG_CSV = "detection_log.csv"
SAVE_DIR = "detected_plates"
SAVE_CROPS = True

# --- LCD (I2C character display) ---
LCD_I2C_ADDRESS = 0x27      # change to match `i2cdetect -y 1` output
LCD_COLS = 16
LCD_ROWS = 2
LCD_PORT = 1                # I2C bus 1 on all modern Raspberry Pi boards

# --- After showing a result, how long to hold it before resuming scanning ---
RESULT_HOLD_SECONDS = 5

# =============================================================================
# PLATE TEXT NORMALIZATION (unchanged from the original implementation)
# =============================================================================

def filter_reading(text):
    """Clean and normalize OCR plate readings for Sri Lankan license plates."""
    if not text:
        return None

    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

    reversed_match = re.match(r'^([0-9]+)([A-Z]+)$', clean_text)
    if reversed_match:
        clean_text = reversed_match.group(2) + reversed_match.group(1)

    numbers = re.search(r'([0-9]{4})$', clean_text)
    if not numbers:
        return None

    main_numbers = numbers.group(1)
    remainder = clean_text.replace(main_numbers, '')

    if re.search(r'[A-Z]', remainder):
        letters = re.sub(r'[0-9]', '', remainder)

        provinces = ['WP', 'SP', 'CP', 'NW', 'NC', 'EP', 'UP', 'SG', 'NP']
        for p in provinces:
            if letters.startswith(p):
                if len(letters) > len(p):
                    letters = letters[len(p):]
                break

        if len(letters) > 3:
            letters = letters[-3:]
        if len(letters) > 2 and letters.startswith("P"):
            letters = letters[-2:]
        if len(letters) > 3:
            letters = letters[:3]

        return f"{letters}-{main_numbers}"

    elif len(remainder) > 0:
        old_numbers = clean_text[-4:]
        old_prefix = clean_text[:-4]
        is_sri = False
        sri_reading = ['S', '3', '8', '0', '2']

        if len(old_prefix) > 2:
            for s in sri_reading:
                if old_prefix.endswith(s):
                    old_prefix = old_prefix[:-1]
                    is_sri = True
                    break

        if is_sri:
            return f"{old_prefix} Sri {old_numbers}"
        else:
            return f"{old_prefix}-{old_numbers}"

    return None

# =============================================================================
# AUTHORIZATION DATABASE (local CSV, replaces the MQTT auth server)
# =============================================================================

class AuthorizationDB:
    """Loads authorized plates from a local CSV file (column: 'plate')."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.plates = set()
        self._ensure_file_exists()
        self.reload()

    def _ensure_file_exists(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["plate"])
                writer.writerow(["WP-1234"])  # sample row, edit/remove freely
            print(f"[AuthDB] Created sample {self.csv_path} - edit it with your plates.")

    def reload(self):
        """Re-read the CSV from disk. Call periodically if you edit it live."""
        plates = set()
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    plate = (row.get("plate") or "").strip().upper()
                    if plate:
                        plates.add(plate)
            self.plates = plates
            print(f"[AuthDB] Loaded {len(self.plates)} authorized plate(s).")
        except Exception as e:
            print(f"[AuthDB] Failed to load {self.csv_path}: {e}")

    def is_authorized(self, plate):
        return plate.strip().upper() in self.plates

# =============================================================================
# I2C LCD DISPLAY
# =============================================================================

class LCDDisplay:
    """Wraps an RPLCD I2C character LCD for status output."""

    def __init__(self, address, cols=16, rows=2, port=1):
        self.cols = cols
        self.rows = rows
        try:
            self.lcd = CharLCD(
                i2c_expander='PCF8574',
                address=address,
                port=port,
                cols=cols,
                rows=rows,
                dotsize=8,
                charmap='A02',
                auto_linebreaks=False,
            )
            self.available = True
        except Exception as e:
            print(f"[LCD] Could not initialize I2C LCD ({e}). "
                  f"Check wiring/address with `i2cdetect -y {port}`.")
            self.lcd = None
            self.available = False
        self.show_idle()

    def _write_lines(self, line1, line2=""):
        if not self.available:
            print(f"[LCD] {line1} | {line2}")
            return
        try:
            self.lcd.clear()
            self.lcd.cursor_pos = (0, 0)
            self.lcd.write_string(line1[:self.cols])
            if self.rows > 1:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2[:self.cols])
        except Exception as e:
            print(f"[LCD] Write failed: {e}")

    def show_idle(self):
        self._write_lines("ANPR Ready", "Scanning...")

    def show_result(self, plate, authorized):
        verdict = "AUTHORIZED" if authorized else "DENIED"
        self._write_lines(plate, verdict)

    def clear(self):
        if self.available:
            try:
                self.lcd.clear()
            except Exception:
                pass

# =============================================================================
# DETECTION LOGGING
# =============================================================================

def log_detection(plate, authorized):
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "parkid", "gate", "direction", "plate", "status"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            PARKID, GATE, DIRECTION, plate,
            "AUTHORIZED" if authorized else "DENIED",
        ])

# =============================================================================
# MAIN ANPR APPLICATION
# =============================================================================

class ANPRApp:
    def __init__(self):
        if SAVE_CROPS and not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        print("Initializing authorization database...")
        self.auth_db = AuthorizationDB(AUTHORIZED_PLATES_CSV)

        print("Initializing LCD...")
        self.lcd = LCDDisplay(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS, LCD_PORT)

        print("Initializing camera (Picamera2)...")
        self.picam2 = Picamera2()
        cam_config = self.picam2.create_video_configuration(
            main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
        )
        # NOTE: Picamera2's "RGB888" format actually yields BGR-ordered
        # arrays (a documented quirk kept for OpenCV compatibility), so
        # frames captured below can be fed straight into cv2 / YOLO as-is.
        self.picam2.configure(cam_config)
        self.picam2.start()
        time.sleep(2)  # let auto-exposure / auto-white-balance settle

        print("Loading YOLO plate-detector model...")
        self.model = YOLO(MODEL_PATH)

        print("Loading EasyOCR reader (this can take a while on a Pi)...")
        self.reader = easyocr.Reader(['en'], gpu=False)

        print("Ready. Scanning for plates. Press Ctrl+C to stop.")

        self.paused_until = 0.0
        self.pending_plate = None
        self.frame_count = 0

        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        print("\nShutting down...")
        self._running = False

    def run(self):
        try:
            while self._running:
                now = time.time()

                # Resume scanning once the result-hold period has elapsed
                if self.pending_plate is not None and now >= self.paused_until:
                    self.pending_plate = None
                    self.lcd.show_idle()

                frame = self.picam2.capture_array()
                self.frame_count += 1

                # While a result is on-screen, skip detection (mirrors the
                # original "inference_paused" behaviour) to save CPU.
                if self.pending_plate is not None:
                    time.sleep(0.05)
                    continue

                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    continue

                self._process_frame(frame)

        finally:
            self.cleanup()

    def _process_frame(self, frame):
        try:
            frame_h, frame_w = frame.shape[:2]
            results = self.model(frame, verbose=False)
            detections = results[0].boxes

            for i in range(len(detections)):
                conf = detections[i].conf.item()
                if conf <= DET_THRESH:
                    continue

                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(frame_w, xmax), min(frame_h, ymax)

                plate_crop = frame[ymin:ymax, xmin:xmax]
                if plate_crop.size == 0:
                    continue

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (1, 1), 10)
                structuring_element = np.zeros((40, 40), np.uint8)
                structuring_element[1:-1, 1:-1] = 1
                final_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)

                ocr_results = self.reader.readtext(
                    final_img, detail=1,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )

                ocr_text = ""
                for (bbox, text, prob) in ocr_results:
                    if prob >= OCR_THRESH:
                        ocr_text += text
                ocr_text = ocr_text.strip()

                if ocr_text and len(ocr_text) >= 6:
                    filtered_plate = filter_reading(ocr_text)
                    if filtered_plate:
                        self._handle_plate(filtered_plate, plate_crop)
                        return  # one plate per cycle, then hold the result

        except Exception as e:
            print(f"Detection error: {e}")

    def _handle_plate(self, plate, plate_crop):
        authorized = self.auth_db.is_authorized(plate)
        verdict = "AUTHORIZED" if authorized else "DENIED"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Plate: {plate} -> {verdict}")

        self.lcd.show_result(plate, authorized)
        log_detection(plate, authorized)

        if SAVE_CROPS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_plate = re.sub(r'[^A-Za-z0-9]', '_', plate)
            fname = os.path.join(SAVE_DIR, f"{ts}_{safe_plate}_{verdict}.jpg")
            try:
                cv2.imwrite(fname, plate_crop)
            except Exception as e:
                print(f"Could not save crop: {e}")

        self.pending_plate = plate
        self.paused_until = time.time() + RESULT_HOLD_SECONDS

    def cleanup(self):
        print("Cleaning up (camera, LCD)...")
        try:
            self.picam2.stop()
        except Exception:
            pass
        self.lcd.clear()
        print("Stopped.")


if __name__ == "__main__":
    app = ANPRApp()
    app.run()
