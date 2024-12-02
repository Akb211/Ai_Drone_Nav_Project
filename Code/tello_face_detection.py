import cv2
from djitellopy import Tello

# Initialize Tello drone
tello = Tello()
tello.connect()
print(f"Battery Life Percentage: {tello.get_battery()}%")

# Start video stream
tello.streamon()

# Initialize OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

try:
    while True:
        # Get video frame from Tello
        frame = tello.get_frame_read().frame

        # Convert the frame to grayscale (required for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the video frame with face detection
        cv2.imshow("Tello Video Stream with Face Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Turn off the video stream and close OpenCV windows
    tello.streamoff()
    cv2.destroyAllWindows()
 