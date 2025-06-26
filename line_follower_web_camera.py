
import cv2
import numpy as np
import time
from collections import deque
from RockyBorg import RockyBorg

# ==== Parameters ====
WIDTH = 320
HEIGHT = 240
ROI_HEIGHT = 80
STEERING_GAIN = 2.0
FORWARD_SPEED = 0.3
LINE_LOST_TIMEOUT = 1.0          # Shortened for more responsive stop
SMOOTHING_WINDOW = 5             # Number of previous centroids to average

# ==== RockyBorg Init ====
RB = RockyBorg()
RB.Init()
RB.SetServoPosition(0.09)
RB.MotorsOff()

# ==== Camera Init ====
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
if not cap.isOpened():
    print("? Cannot open camera.")
    exit()

print("? Ready. Waiting for black line...")

# ==== State Variables ====
centroid_history = deque(maxlen=SMOOTHING_WINDOW)
motors_on = False
last_seen_time = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("? Camera error")
            break

        current_time = time.time()

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        roi = binary[HEIGHT - ROI_HEIGHT:, :]

        # Optional: crop center region (wider for curves)
        center_band_width = WIDTH // 2     # Widened for better curve tracking
        x_start = WIDTH // 2 - center_band_width // 2
        x_end = WIDTH // 2 + center_band_width // 2
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
                    error = smoothed_cx - (WIDTH // 2)
                    steering = max(-1.0, min(1.0, (error / (WIDTH // 2)) * STEERING_GAIN))
                    RB.SetServoPosition(steering)

                    # Start motors
                    if not motors_on:
                        RB.SetMotor1(-FORWARD_SPEED)
                        RB.SetMotor2(FORWARD_SPEED)
                        motors_on = True
                        print("? Line detected ? motors running")

                    last_seen_time = current_time
                    found_line = True

                    # Visual feedback
                    cy = HEIGHT - ROI_HEIGHT + 5
                    cv2.drawContours(frame[HEIGHT - ROI_HEIGHT:], [largest], -1, (0, 0, 255), 2)
                    cv2.circle(frame, (smoothed_cx, cy), 5, (0, 255, 0), -1)
                    cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (255, 0, 0), 1)
        
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
        cv2.imshow("ROI", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("? Exiting...")
            break

except KeyboardInterrupt:
    print("? Interrupted by user")

# ==== Cleanup ====
RB.MotorsOff()
cap.release()
cv2.destroyAllWindows()
