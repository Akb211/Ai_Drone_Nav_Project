import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from djitellopy import Tello
import torch


class HumanDetector:
    def __init__(self, model_path, device="cuda:0"):
        # Verify the model file exists locally
        if not os.path.isfile(model_path):
            print(f"Error: YOLOv11 model file '{model_path}' not found.")
            exit(1)
        try:
            self.model = YOLO(model_path)
            self.model.to(device)  # Move model to GPU if available
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            exit(1)
        self.names = self.model.names  # Class names
        self.frame_skip_count = 0
        self.max_frame_skip = 3 
    def detect(self, frame, conf_threshold=0.5):
        results = self.model(frame, verbose=False)[0]  # Perform detection
        boxes = []
        for box in results.boxes:
            conf = box.conf.item()  # Confidence score
            cls = int(box.cls.item())  # Class ID
            label = self.names.get(cls, None)  # Class label
            if label == "person" and conf >= conf_threshold:
                # Convert bounding box coordinates to [x1, y1, w, h]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
        return boxes
class OptimizedHumanDetector(HumanDetector):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        # Enable memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            
    def detect(self, frame, conf_threshold=0.5):
        with torch.cuda.amp.autocast():  # Mixed precision for speed
            return super().detect(frame, conf_threshold)
class HumanTracker:
    def __init__(self):
        self.trackers = []  # List to store trackers

    def init_trackers(self, frame, boxes):
        self.trackers = []
        for box in boxes:
            try:
                tracker = cv2.TrackerKCF_create()  # Use a faster tracker
                tracker.init(frame, tuple(box))
                self.trackers.append((tracker, box))
            except Exception as e:
                print(f"Error initializing tracker: {e}")
                continue

    def update(self, frame):
        updated_boxes = []
        new_trackers = []
        for tracker, _ in self.trackers:
            success, box = tracker.update(frame)
            if success:
                updated_boxes.append(box)
                new_trackers.append((tracker, box))
        self.trackers = new_trackers
        return updated_boxes
    def detect_with_optimization(self, frame, conf_threshold=0.5):
        # Skip frames intelligently based on detection state
        if self.frame_skip_count > 0:
            self.frame_skip_count -= 1
            return []  # Return empty if skipping
            
        boxes = self.detect(frame, conf_threshold)
        
        # Adjust frame skipping based on detection results
        if len(boxes) == 0:
            self.frame_skip_count = self.max_frame_skip
        else:
            self.frame_skip_count = 0  # Process every frame when humans detected
            
        return boxes

def main():
    # Connect to the Tello EDU drone
    tello = Tello()
    try:
        tello.connect()
        print("Tello Battery:", tello.get_battery())
    except Exception as e:
        print(f"Error connecting to Tello: {e}")
        exit(1)

    # Start video stream with a delay
    tello.streamon()
    time.sleep(2)  # Wait for the stream to initialize
    frame_read = tello.get_frame_read()

    # Initialize detector and tracker
    detector = HumanDetector("yolo11n.pt", device="cuda:0")  # Load YOLOv11 model
    tracker = HumanTracker()

    detection_interval = 10  # Run detection every 10 frames
    frame_count = 0
    tracked_boxes = []

    # Initialize FPS calculation
    prev_time = time.time()

    try:
        while True:
            # Read frame from the Tello drone
            frame = frame_read.frame
            if frame is None or frame.size == 0:
                print("Invalid frame received.")
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (320, 240))

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_time = curr_time

            frame_count += 1
            print(f"Processing frame {frame_count} (FPS: {fps:.2f})")

            # Run detection every detection_interval frames, otherwise update trackers
            if frame_count % detection_interval == 0:
                print("Running YOLOv11 detection...")
                boxes = detector.detect(frame)
                if boxes:  # Only initialize trackers if boxes are detected
                    tracked_boxes = boxes
                    tracker.init_trackers(frame, boxes)
            else:
                boxes = tracker.update(frame)
                tracked_boxes = boxes

            # Draw bounding boxes and labels
            for box in tracked_boxes:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Tello YOLOv11 Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up resources
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
