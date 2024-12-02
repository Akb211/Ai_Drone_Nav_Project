import cv2
import numpy as np
from djitellopy import Tello
import time

# PID controller constants
Kp = 0.5
Ki = 0.1
Kd = 0.2

# PID variables
previous_error = 0
integral = 0

def pid_control(error, previous_error, integral):
    """
    PID controller to calculate adjustment for drone motion.
    """
    global Kp, Ki, Kd
    integral += error
    derivative = error - previous_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

# Initialize Tello drone
tello = Tello()
tello.connect()
print(f"Battery Life Percentage: {tello.get_battery()}%")

# Start video stream
tello.streamon()

# Take off
tello.takeoff()
time.sleep(2)

# Set up OpenCV windows
cv2.namedWindow("Tello Obstacle Avoidance")

try:
    while True:
        # Capture frame from Tello's camera
        frame = tello.get_frame_read().frame

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Convert frame to HSV for color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for detecting red obstacles
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Range for red (upper spectrum)
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine both masks
        mask = mask1 + mask2

        # Find contours of the red object (obstacle)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize the center of the obstacle
        obstacle_center = None

        # If an obstacle is detected
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(largest_contour)

            # Calculate the center of the obstacle
            obstacle_center = (x + w // 2, y + h // 2)

            # Draw a rectangle around the detected obstacle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, obstacle_center, 5, (255, 0, 0), -1)

            # Calculate error (distance from center of the frame)
            frame_center_x = frame.shape[1] // 2
            error = frame_center_x - obstacle_center[0]

            # Apply PID control to calculate movement adjustment
            adjustment, integral = pid_control(error, previous_error, integral)

            # Move the drone to avoid the obstacle
            if abs(error) > 20:  # Threshold to avoid overcorrection
                if error > 0:
                    tello.move_left(int(min(abs(adjustment), 20)))  # Move left
                else:
                    tello.move_right(int(min(abs(adjustment), 20)))  # Move right

            # Update the previous error
            previous_error = error

        # Display the video feed
        cv2.imshow("Tello Obstacle Avoidance", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Land the drone safely
    tello.land()

except KeyboardInterrupt:
    print("Emergency stop! Landing...")
    tello.land()

finally:
    # Turn off the video stream and close OpenCV windows
    tello.streamoff()
    cv2.destroyAllWindows()
