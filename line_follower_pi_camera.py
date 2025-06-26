import cv2
import numpy as np
import time
from collections import deque
from picamera2 import Picamera2
from libcamera import Transform
from RockyBorg import RockyBorg

# ==== Parameters ====
WIDTH = 640
HEIGHT = 640
ROI_HEIGHT = 640
STEERING_GAIN = 1.0
FORWARD_SPEED = 0.25
LINE_LOST_TIMEOUT = 1.0
SMOOTHING_WINDOW = 5

# ==== RockyBorg Init ====
RB = RockyBorg()
RB.Init()
RB.SetServoPosition(0.09)
RB.MotorsOff()

# ==== PiCamera2 Setup ====
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (WIDTH, HEIGHT)}))
picam2.start()
time.sleep(2)

print("? Ready. Waiting for black line...")

# ==== State Variables ====
centroid_history = deque(maxlen=SMOOTHING_WINDOW)
motors_on = False
last_seen_time = None
previous_steering=0.0
STEERING_SMOOTHING=0.2

try:
    while True:
        frame = picam2.capture_array()

        # Rotate frame to match actual orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        current_time = time.time()

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        roi = binary[frame.shape[0] - ROI_HEIGHT:, :]

        # Crop center band for better curve following
        center_band_width = frame.shape[1] // 2
        x_start = frame.shape[1] // 2 - center_band_width // 2
        x_end = frame.shape[1] // 2 + center_band_width // 2
        roi_mask = np.zeros_like(roi)
        roi_mask[:, x_start:x_end] = roi[:, x_start:x_end]
        roi = roi_mask

        # Contour detection
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_line = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    centroid_history.append(cx)
                    smoothed_cx = int(np.mean(centroid_history))

                    # Steering control
                    error = smoothed_cx - (frame.shape[1] // 2)
                    raw_steering = (error / (WIDTH // 2)) * STEERING_GAIN
                    steering = (1 - STEERING_SMOOTHING) * previous_steering + STEERING_SMOOTHING * raw_steering
                    steering = max(-1.0, min(1.0, steering))
                    RB.SetServoPosition(steering)
                    previous_steering = steering

                    # Start motors if not already running
                    if not motors_on:
                        RB.SetMotor1(-FORWARD_SPEED)
                        RB.SetMotor2(FORWARD_SPEED)
                        motors_on = True
                        print("? Line detected ? motors running")

                    last_seen_time = current_time
                    found_line = True

                    # Visual feedback
                    cy = frame.shape[0] - ROI_HEIGHT + 5
                    cv2.drawContours(frame[frame.shape[0] - ROI_HEIGHT:], [largest], -1, (0, 0, 255), 2)
                    cv2.circle(frame, (smoothed_cx, cy), 5, (0, 255, 0), -1)
                    cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (255, 0, 0), 1)

        if not found_line:
            if last_seen_time is None:
                last_seen_time = current_time

            time_since_seen = current_time - last_seen_time
            if time_since_seen > LINE_LOST_TIMEOUT and motors_on:
                RB.MotorsOff()
                motors_on = False
                print(f"? Line lost for {LINE_LOST_TIMEOUT}s ? motors stopped")

        # Show output
        cv2.imshow("View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("? Exiting...")
            break

except KeyboardInterrupt:
    print("? Interrupted by user")

# ==== Cleanup ====
RB.MotorsOff()
cv2.destroyAllWindows()
