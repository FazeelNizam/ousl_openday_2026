#!/usr/bin/env python3
"""
==============================================================================
 Headless ANPR (Automatic Number Plate Recognition) for Raspberry Pi 4B
 Camera:  Raspberry Pi Camera Module V1.3 (OV5647) via Picamera2/libcamera
 Output:  Parallel HD44780 character LCD wired directly to GPIO (no I2C
          backpack), showing the plate and AUTHORIZED / DENIED
 Gate:    360-degree continuous-rotation 9g servo, timed open/close
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
 - Added:   Direct-GPIO parallel character LCD output (RPLCD's GPIO
   backend, HD44780-compatible) showing the plate and AUTHORIZED / DENIED.
 - Added:   Gate control for a 360-degree continuous-rotation servo -
   rotates open, holds, then rotates closed on an AUTHORIZED result.
 - Kept:    The YOLO (ncnn) plate-detector + EasyOCR reading pipeline and the
   `filter_reading()` plate-normalization logic, unchanged, since that part
   is hardware-independent.

WIRING - LCD (parallel HD44780, 8-bit data bus)
------------------------------------------------------------------------------
 This is a plain character LCD (16-pin header: VSS, VDD, V0, RS, RW, E,
 D0-D7, A/K for the backlight) - NOT an I2C module, so it's wired straight
 to GPIO pins:
   LCD pin   -> Pi
   VSS       -> GND
   VDD       -> 5V
   V0        -> wiper of a 10k potentiometer between 5V and GND (contrast)
   RS        -> BCM 5   (LCD_PIN_RS below)
   RW        -> GND     (tie low; this script only ever writes, never reads)
   E         -> BCM 6   (LCD_PIN_E below)
   D0-D7     -> BCM 12,13,16,19,20,21,26,7  (LCD_DATA_PINS below, in order)
   A (LED+)  -> 5V (through a ~220ohm resistor if your board doesn't have one)
   K (LED-)  -> GND
 You already have all 8 data lines wired, so this script uses 8-bit mode by
 default. If you'd rather free up 4 GPIO pins, you can rewire to only
 D4-D7 and set LCD_DATA_PINS to those 4 pins (RPLCD auto-switches to 4-bit
 mode based on how many pins you give it) - leave D0-D3 disconnected.

WIRING - Gate servo (360-degree continuous-rotation, 9g)
------------------------------------------------------------------------------
 - Servo signal wire -> BCM 18 (SERVO_PIN below; this pin has hardware PWM)
 - Servo power (red)  -> an external 5V supply, NOT the Pi's 5V pin -
   a small 9g servo can spike enough current to brown out the Pi.
 - Servo ground (black/brown) -> common GND with the Pi (must be shared).
 A continuous-rotation servo has no fixed "angle" - it just spins CW, CCW,
 or stops depending on the pulse width, so there's no absolute open/closed
 position to read back. This script treats the gate as timed motion: spin
 one way for GATE_OPEN_DURATION seconds to swing the arm open, hold it open
 for GATE_HOLD_OPEN_SECONDS, then spin the other way for
 GATE_CLOSE_DURATION seconds to swing it back. Time these against your
 actual gate arm - start short and increase until it swings a full cycle.

ONE-TIME RASPBERRY PI SETUP
------------------------------------------------------------------------------
1. Enable the camera interface:
       sudo raspi-config
       -> Interface Options -> Camera -> Enable
   Reboot afterwards. (I2C is no longer needed for this LCD.)

2. (Optional, for smoother servo motion) The pigpio daemon gives
   hardware-timed PWM instead of software PWM. It has been dropped from
   the apt repos on current Raspberry Pi OS (Bookworm and later), since it
   doesn't support the Pi 5. If `sudo apt install pigpio python3-pigpio`
   fails with "Unable to locate package", just skip it - the script
   detects this and falls back to gpiozero's default pin factory
   automatically, which is fine for a slow gate open/close motion. If you
   do want it and your apt has it available:
       sudo apt install -y pigpio python3-pigpio
       sudo systemctl enable pigpiod
       sudo systemctl start pigpiod

3. System packages:
       sudo apt update
       sudo apt install -y python3-picamera2 --no-install-recommends \
                            python3-pip python3-opencv python3-rpi.gpio

4. Python packages (a virtualenv with --system-site-packages is recommended
   so it can see the apt-installed picamera2 and RPi.GPIO):
       python3 -m venv --system-site-packages venv
       source venv/bin/activate
       pip install ultralytics easyocr RPLCD gpiozero
   (Only add `pip install pigpio` if you got the optional pigpio daemon
   running in step 2 above - otherwise skip it, it's not required.)

   IMPORTANT - opencv-python vs opencv-python-headless: installing
   ultralytics pulls in the GUI-enabled `opencv-python` package as a
   dependency automatically, which needs libGL.so.1 and other X11/GTK
   libraries that a headless Raspberry Pi OS Lite install typically does
   NOT have - you'll get "ImportError: libGL.so.1: cannot open shared
   object file" the moment this script tries to `import cv2`. Force the
   headless build instead, right after installing the packages above:
       pip uninstall -y opencv-python opencv-contrib-python
       pip install opencv-python-headless
   (This is safe even though we also apt-installed python3-opencv earlier -
   whichever cv2 is actually present in the venv's site-packages at import
   time is the one Python uses, so this just makes sure it's the headless
   one and not the GUI one ultralytics pulled in.)

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

7. Adjust LCD_PIN_RS / LCD_PIN_E / LCD_DATA_PINS and SERVO_PIN below to
   match however you actually wired things, and calibrate
   GATE_OPEN_DURATION / GATE_CLOSE_DURATION by testing.

RUN
------------------------------------------------------------------------------
       python3 anpr_headless.py
   Stop with Ctrl+C (the LCD is cleared and the servo is released on exit).
==============================================================================
"""

import os
import re
import csv
import time
import signal
import sys
import threading
from datetime import datetime

import cv2
import numpy as np

from picamera2 import Picamera2
from ultralytics import YOLO
import easyocr
import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD

from gpiozero import Servo

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

# --- LCD (parallel HD44780, wired directly to GPIO - BCM numbering) ---
LCD_COLS = 16
LCD_ROWS = 2
LCD_PIN_RS = 5
LCD_PIN_E = 6
# 8 entries = 8-bit mode (D0-D7), matching your existing wiring.
# To switch to 4-bit mode instead, wire only D4-D7 and give just 4 pins here.
LCD_DATA_PINS = [12, 13, 16, 19, 20, 21, 26, 7]   # D0..D7
LCD_PIN_BACKLIGHT = None    # set a BCM pin here only if backlight is switched via GPIO

# --- Gate servo (360-degree continuous-rotation, timed open/close) ---
SERVO_PIN = 18                  # BCM, hardware-PWM-capable pin
SERVO_MIN_PULSE_WIDTH = 0.001   # 1.0 ms  - full speed one direction
SERVO_MAX_PULSE_WIDTH = 0.002   # 2.0 ms  - full speed other direction
GATE_OPEN_SPEED = 1.0           # -1.0..1.0, direction/speed to open the gate
GATE_CLOSE_SPEED = -1.0         # opposite direction to close it
GATE_OPEN_DURATION = 1.0        # seconds of rotation to swing fully open - CALIBRATE THIS
GATE_CLOSE_DURATION = 1.0       # seconds of rotation to swing fully closed - CALIBRATE THIS
GATE_HOLD_OPEN_SECONDS = 4.0    # how long to keep the gate open before auto-closing

# --- After showing a result, how long to hold it before resuming scanning ---
# For an AUTHORIZED result this should comfortably cover the full gate cycle
# (open + hold + close); it's recalculated per-event in _handle_plate() too.
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
# PARALLEL (GPIO) LCD DISPLAY
# =============================================================================

class LCDDisplay:
    """Wraps an RPLCD GPIO-driven HD44780 character LCD for status output."""

    def __init__(self, pin_rs, pin_e, data_pins, cols=16, rows=2, pin_backlight=None):
        self.cols = cols
        self.rows = rows
        try:
            self.lcd = CharLCD(
                pin_rs=pin_rs,
                pin_rw=None,          # RW tied to GND on the LCD (write-only)
                pin_e=pin_e,
                pins_data=data_pins,  # 4 pins = 4-bit mode, 8 pins = 8-bit mode
                pin_backlight=pin_backlight,
                numbering_mode=GPIO.BCM,
                cols=cols,
                rows=rows,
                dotsize=8,
                charmap='A02',
                auto_linebreaks=False,
            )
            self.available = True
        except Exception as e:
            print(f"[LCD] Could not initialize GPIO LCD ({e}). "
                  f"Check the RS/E/data-pin wiring and numbers above.")
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
# GATE CONTROL (360-degree continuous-rotation servo)
# =============================================================================

class GateController:
    """
    Drives a continuous-rotation servo as a timed park-gate actuator.

    Continuous-rotation servos have no absolute position feedback - the PWM
    value just sets speed/direction (-1..1, 0 = stop). So "open" and "close"
    here are timed rotations, not angles. Calibrate GATE_OPEN_DURATION /
    GATE_CLOSE_DURATION against your physical gate arm.
    """

    def __init__(self, pin, min_pulse_width, max_pulse_width):
        # pigpio (if installed and pigpiod is running) gives hardware-timed
        # pulses -> smoother servo motion than the default software PWM pin
        # factory. It's entirely optional and often unavailable on current
        # Raspberry Pi OS (Bookworm+ dropped it from apt), so this import
        # and connection attempt are both wrapped in try/except - if either
        # fails, gpiozero's default pin factory (lgpio or RPi.GPIO) is used
        # instead, which works fine for a slow open/close gate motion.
        factory = None
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory
            factory = PiGPIOFactory()
        except Exception as e:
            print(f"[Gate] pigpio not available ({e}); using the default "
                  f"software-PWM pin factory instead. This is fine for a "
                  f"gate servo - install/run pigpiod later if motion is jittery.")
            factory = None

        try:
            self.servo = Servo(
                pin,
                min_pulse_width=min_pulse_width,
                max_pulse_width=max_pulse_width,
                pin_factory=factory,
            )
            self.servo.value = None  # stop sending pulses (avoids buzzing/holding torque)
            self.available = True
        except Exception as e:
            print(f"[Gate] Could not initialize servo on pin {pin} ({e}).")
            self.servo = None
            self.available = False

        self._lock = threading.Lock()

    def _spin(self, speed, duration):
        if not self.available:
            print(f"[Gate] (simulated) spin at {speed} for {duration}s")
            time.sleep(duration)
            return
        self.servo.value = speed
        time.sleep(duration)
        self.servo.value = None  # stop / release - avoids the servo humming continuously

    def open_gate(self):
        with self._lock:
            print("[Gate] Opening...")
            self._spin(GATE_OPEN_SPEED, GATE_OPEN_DURATION)

    def close_gate(self):
        with self._lock:
            print("[Gate] Closing...")
            self._spin(GATE_CLOSE_SPEED, GATE_CLOSE_DURATION)

    def cycle_open_hold_close(self, hold_seconds):
        """Open, wait, then close - run this in a background thread."""
        self.open_gate()
        time.sleep(hold_seconds)
        self.close_gate()

    def release(self):
        if self.available:
            try:
                self.servo.value = None
                self.servo.close()
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
        self.lcd = LCDDisplay(LCD_PIN_RS, LCD_PIN_E, LCD_DATA_PINS,
                               LCD_COLS, LCD_ROWS, LCD_PIN_BACKLIGHT)

        print("Initializing gate servo...")
        self.gate = GateController(SERVO_PIN, SERVO_MIN_PULSE_WIDTH, SERVO_MAX_PULSE_WIDTH)

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

        if authorized:
            # Run the gate cycle (open -> hold -> close) in the background so
            # the camera loop's pause timer and the physical gate stay in
            # sync without blocking signal handling.
            gate_total = GATE_OPEN_DURATION + GATE_HOLD_OPEN_SECONDS + GATE_CLOSE_DURATION
            self.paused_until = time.time() + max(RESULT_HOLD_SECONDS, gate_total)
            threading.Thread(
                target=self.gate.cycle_open_hold_close,
                args=(GATE_HOLD_OPEN_SECONDS,),
                daemon=True,
            ).start()
        else:
            self.paused_until = time.time() + RESULT_HOLD_SECONDS

    def cleanup(self):
        print("Cleaning up (camera, LCD, gate)...")
        try:
            self.picam2.stop()
        except Exception:
            pass
        self.lcd.clear()
        self.gate.release()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        print("Stopped.")


if __name__ == "__main__":
    app = ANPRApp()
    app.run()
