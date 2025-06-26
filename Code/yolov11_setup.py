# update_existing_code.py
"""
Update your existing tello_yolo_detection.py to work with downloaded YOLOv11 models
"""

def update_human_detector():
    """Update the HumanDetector class with better model handling"""
    
    updated_code = '''
# Updated tello_yolo_detection.py
import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
from djitellopy import Tello
import torch

class HumanDetector:
    def __init__(self, model_path="models/yolo11n.pt", device="auto"):
        """
        Initialize YOLOv11 Human Detector
        
        Args:
            model_path: Path to YOLOv11 model (will download if needed)
            device: Device to run on ("auto", "cpu", "cuda", "mps")
        """
        
        # Auto-select best available device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print("🚀 Using GPU acceleration")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                print("🍎 Using Apple Silicon acceleration")
            else:
                device = "cpu"
                print("💻 Using CPU")
        
        self.device = device
        
        # Handle model path and auto-download
        if not os.path.isfile(model_path):
            print(f"⬇️ Model not found at {model_path}, downloading...")
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Extract model name from path
            model_name = os.path.basename(model_path)
            
            try:
                # YOLO will auto-download if model doesn't exist
                self.model = YOLO(model_name)
                
                # Move to correct location
                if os.path.exists(model_name):
                    os.rename(model_name, model_path)
                    print(f"✅ Downloaded and saved to {model_path}")
                
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                print("🔄 Trying with yolo11n.pt as fallback...")
                self.model = YOLO("yolo11n.pt")
        else:
            try:
                self.model = YOLO(model_path)
                print(f"✅ Loaded model from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
        
        # Move model to selected device
        self.model.to(device)
        
        # Get class names
        self.names = self.model.names
        
        # Performance optimization
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
        print(f"🎯 Human detector ready on {device}")

    def detect(self, frame, conf_threshold=0.4):
        """
        Detect humans in frame with underground optimizations
        
        Args:
            frame: Input image
            conf_threshold: Confidence threshold (lowered for underground)
            
        Returns:
            List of bounding boxes [x1, y1, w, h]
        """
        start_time = time.time()
        
        try:
            # Enhanced preprocessing for underground conditions
            enhanced_frame = self._enhance_for_underground(frame)
            
            # Run inference with mixed precision for better performance
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    results = self.model(enhanced_frame, verbose=False)[0]
            else:
                results = self.model(enhanced_frame, verbose=False)[0]
            
            # Track performance
            inference_time = time.time() - start_time
            self.total_time += inference_time
            self.frame_count += 1
            
            # Extract person detections
            boxes = []
            if results.boxes is not None:
                for box in results.boxes:
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = self.names.get(cls, None)
                    
                    # Adaptive threshold for low-light conditions
                    adaptive_threshold = self._get_adaptive_threshold(frame, conf_threshold)
                    
                    if label == "person" and conf >= adaptive_threshold:
                        # Convert bounding box coordinates to [x1, y1, w, h]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
            
            return boxes
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _enhance_for_underground(self, frame):
        """Enhance image for better detection in underground conditions"""
        
        # Convert to LAB color space for better low-light enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to improve contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional brightness boost for very dark images
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if brightness < 50:  # Very dark
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
        
        return enhanced
    
    def _get_adaptive_threshold(self, frame, base_threshold):
        """Get adaptive confidence threshold based on lighting conditions"""
        
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Lower threshold for darker conditions
        if brightness < 30:  # Very dark
            return base_threshold * 0.6
        elif brightness < 60:  # Dark
            return base_threshold * 0.8
        else:  # Normal/bright
            return base_threshold
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.frame_count == 0:
            return {"avg_fps": 0, "avg_time": 0}
        
        avg_time = self.total_time / self.frame_count
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            "avg_fps": avg_fps,
            "avg_time": avg_time,
            "total_frames": self.frame_count,
            "device": self.device
        }

class HumanTracker:
    def __init__(self):
        self.trackers = []  # List to store trackers
        self.tracker_ids = []  # Track IDs for consistency
        self.next_id = 0

    def init_trackers(self, frame, boxes):
        """Initialize trackers for detected humans"""
        self.trackers = []
        self.tracker_ids = []
        
        for box in boxes:
            try:
                # Use KCF tracker (fast and reliable)
                tracker = cv2.TrackerKCF_create()
                success = tracker.init(frame, tuple(box))
                
                if success:
                    self.trackers.append(tracker)
                    self.tracker_ids.append(self.next_id)
                    self.next_id += 1
                    
            except Exception as e:
                print(f"Error initializing tracker: {e}")
                continue

    def update(self, frame):
        """Update all trackers and return updated bounding boxes"""
        updated_boxes = []
        valid_trackers = []
        valid_ids = []
        
        for i, tracker in enumerate(self.trackers):
            try:
                success, box = tracker.update(frame)
                if success:
                    # Validate box coordinates
                    x, y, w, h = box
                    if w > 10 and h > 10:  # Minimum size filter
                        updated_boxes.append([int(x), int(y), int(w), int(h)])
                        valid_trackers.append(tracker)
                        valid_ids.append(self.tracker_ids[i])
                        
            except Exception as e:
                print(f"Tracker update error: {e}")
                continue
        
        # Update tracker lists with only valid trackers
        self.trackers = valid_trackers
        self.tracker_ids = valid_ids
        
        return updated_boxes

def main():
    """Updated main function with better error handling and model management"""
    
    print("🚁 Underground Drone System - Human Detection")
    print("=" * 50)
    
    # Connect to the Tello EDU drone
    tello = Tello()
    try:
        tello.connect()
        battery = tello.get_battery()
        print(f"🔋 Tello Battery: {battery}%")
        
        if battery < 20:
            print("⚠️ Warning: Low battery! Consider charging before flight.")
            
    except Exception as e:
        print(f"❌ Error connecting to Tello: {e}")
        print("🔄 Running in demo mode with test images...")
        tello = None

    # Initialize detector with auto-download
    print("🧠 Initializing YOLOv11 detector...")
    try:
        detector = HumanDetector("models/yolo11n.pt", device="auto")
        tracker = HumanTracker()
        
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return

    # Start video stream if drone is available
    if tello:
        try:
            tello.streamon()
            time.sleep(2)  # Wait for stream initialization
            frame_read = tello.get_frame_read()
            print("📹 Video stream started")
        except Exception as e:
            print(f"❌ Video stream error: {e}")
            return
    else:
        # Demo mode - use webcam or test images
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No camera available for demo mode")
            return
        print("📹 Using webcam for demo")

    detection_interval = 10  # Run detection every 10 frames
    frame_count = 0
    tracked_boxes = []
    
    print("🚀 Starting detection loop...")
    print("Press 'q' to quit, 's' to show stats")

    try:
        while True:
            # Get frame
            if tello:
                frame = frame_read.frame
                if frame is None or frame.size == 0:
                    print("⚠️ Invalid frame received")
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame")
                    break

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            # Run detection or update trackers
            if frame_count % detection_interval == 0:
                print(f"🔍 Running detection on frame {frame_count}")
                boxes = detector.detect(frame, conf_threshold=0.4)
                
                if boxes:
                    tracked_boxes = boxes
                    tracker.init_trackers(frame, boxes)
                    print(f"👥 Detected {len(boxes)} person(s)")
                else:
                    print("👤 No persons detected")
            else:
                # Update trackers
                if tracker.trackers:
                    tracked_boxes = tracker.update(frame)

            # Draw bounding boxes and info
            for i, box in enumerate(tracked_boxes):
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add tracker ID if available
                if i < len(tracker.tracker_ids):
                    label = f"Person {tracker.tracker_ids[i]}"
                else:
                    label = "Person"
                    
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add performance info
            stats = detector.get_performance_stats()
            info_text = f"FPS: {stats['avg_fps']:.1f} | Frame: {frame_count}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display frame
            cv2.imshow("Underground Drone - Human Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                stats = detector.get_performance_stats()
                print(f"📊 Performance: {stats}")

    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
        
    finally:
        # Cleanup
        if tello:
            try:
                tello.streamoff()
                tello.end()
            except:
                pass
        else:
            cap.release()
            
        cv2.destroyAllWindows()
        
        # Final stats
        final_stats = detector.get_performance_stats()
        print(f"\\n📈 Final Performance Stats: {final_stats}")
        print("✅ Detection system shutdown complete")

if __name__ == "__main__":
    main()
'''
    
    # Write the updated code to a new file
    with open("tello_yolo_detection_updated.py", "w") as f:
        f.write(updated_code)
    
    print("✅ Created tello_yolo_detection_updated.py with YOLOv11 support")

def update_main_py():
    """Create updated main.py snippet for better model handling"""
    
    main_update = '''
# Add this to your main.py for better model handling

def parse_args():
    parser = argparse.ArgumentParser(description='Underground Drone System with Human Detection, SLAM, and RL')
    
    # Updated model argument with auto-download
    parser.add_argument('--model', type=str, default='models/yolo11n.pt', 
                      help='Path to YOLOv11 model (will auto-download if not found)')
    parser.add_argument('--device', type=str, default='auto', 
                      help='Device to run YOLO model on (auto/cpu/cuda/mps)')
    
    # Add model performance options
    parser.add_argument('--detection-confidence', type=float, default=0.4,
                      help='Detection confidence threshold (lower for underground)')
    parser.add_argument('--enhance-underground', action='store_true',
                      help='Enable image enhancement for underground conditions')
    
    # ... rest of your arguments
    
    return parser.parse_args()

def setup_detector_with_fallback(args):
    """Setup detector with automatic model downloading and fallback"""
    
    print("🧠 Setting up YOLOv11 detector...")
    
    # Try primary model first
    try:
        from tello_yolo_detection_updated import HumanDetector
        detector = HumanDetector(args.model, device=args.device)
        print(f"✅ Using {args.model}")
        return detector
        
    except Exception as e:
        print(f"⚠️ Failed to load {args.model}: {e}")
        
        # Fallback to nano model
        fallback_models = ['models/yolo11n.pt', 'yolo11n.pt']
        
        for fallback in fallback_models:
            try:
                print(f"🔄 Trying fallback: {fallback}")
                detector = HumanDetector(fallback, device=args.device)
                print(f"✅ Using fallback model: {fallback}")
                return detector
                
            except Exception as e:
                print(f"❌ Fallback {fallback} failed: {e}")
                continue
        
        raise RuntimeError("❌ Could not initialize any YOLOv11 model")

# In your main() function, replace detector initialization:
def main():
    args = parse_args()
    
    # ... other setup code ...
    
    # Setup detector with auto-download and fallbacks
    detector = setup_detector_with_fallback(args)
    
    # ... rest of your code ...
'''
    
    with open("main_py_updates.py", "w") as f:
        f.write(main_update)
    
    print("✅ Created main_py_updates.py with model handling improvements")

def create_quick_test_script():
    """Create a quick test script to verify everything works"""
    
    test_script = '''
#!/usr/bin/env python3
# quick_test.py - Test YOLOv11 setup
"""
Quick test script to verify YOLOv11 setup is working correctly
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def test_model_download():
    """Test if models can be downloaded and loaded"""
    print("🧪 Testing YOLOv11 model download and loading...")
    
    try:
        from ultralytics import YOLO
        
        # Test with nano model (smallest/fastest)
        model = YOLO("yolo11n.pt")
        print("✅ YOLOv11n loaded successfully")
        
        # Test inference on dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print("✅ Inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_optimized_detector():
    """Test the optimized underground detector"""
    print("\\n🧪 Testing optimized underground detector...")
    
    try:
        # Import our optimized detector
        if os.path.exists("optimized_underground_detector.py"):
            sys.path.append(".")
            from optimized_underground_detector import OptimizedUndergroundDetector
            
            detector = OptimizedUndergroundDetector("yolo11n.pt")
            print("✅ Optimized detector loaded")
            
            # Test with synthetic underground image
            test_img = create_underground_test_image()
            detections = detector.detect(test_img)
            
            print(f"✅ Detection test: {len(detections)} detections")
            print(f"📊 Performance: {detector.get_performance_stats()}")
            
            return True
            
        else:
            print("⚠️ optimized_underground_detector.py not found")
            return False
            
    except Exception as e:
        print(f"❌ Optimized detector test failed: {e}")
        return False

def create_underground_test_image():
    """Create a test image with underground conditions and a person"""
    
    # Dark underground environment
    img = np.ones((480, 640, 3), dtype=np.uint8) * 25
    
    # Add tunnel structure
    cv2.rectangle(img, (0, 0), (640, 80), (40, 40, 40), -1)      # Ceiling
    cv2.rectangle(img, (0, 400), (640, 480), (40, 40, 40), -1)   # Floor
    cv2.rectangle(img, (0, 0), (30, 480), (40, 40, 40), -1)      # Left wall
    cv2.rectangle(img, (610, 0), (640, 480), (40, 40, 40), -1)   # Right wall
    
    # Add person figure
    person_color = (80, 80, 80)
    
    # Head
    cv2.circle(img, (320, 160), 18, person_color, -1)
    
    # Body  
    cv2.rectangle(img, (305, 178), (335, 280), person_color, -1)
    
    # Arms
    cv2.rectangle(img, (285, 190), (305, 250), person_color, -1)
    cv2.rectangle(img, (335, 190), (355, 250), person_color, -1)
    
    # Legs
    cv2.rectangle(img, (310, 280), (325, 350), person_color, -1)
    cv2.rectangle(img, (325, 280), (340, 350), person_color, -1)
    
    # Add some noise for realism
    noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def test_integration():
    """Test integration with existing code"""
    print("\\n🧪 Testing integration with existing system...")
    
    try:
        # Test imports
        modules_to_test = [
            "tello_yolo_detection_updated",
            "collision", 
            "rl"
        ]
        
        for module in modules_to_test:
            if os.path.exists(f"{module}.py"):
                try:
                    __import__(module)
                    print(f"✅ {module}.py imports successfully")
                except Exception as e:
                    print(f"⚠️ {module}.py import issue: {e}")
            else:
                print(f"ℹ️ {module}.py not found (optional)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def display_recommendations():
    """Display recommendations based on test results"""
    print("\\n" + "="*50)
    print("🎯 RECOMMENDATIONS")
    print("="*50)
    
    print("✅ Quick Start Commands:")
    print("   python yolov11_setup.py                    # Download all models")
    print("   python model_comparison.py                 # Compare performance")
    print("   python tello_yolo_detection_updated.py     # Test detection")
    print("   python main.py --model models/yolo11n.pt   # Run full system")
    
    print("\\n⚡ For Best Performance:")
    print("   • Use yolo11n.pt for real-time operation")
    print("   • Use GPU if available (CUDA/MPS)")
    print("   • Enable underground image enhancement")
    print("   • Set confidence threshold to 0.4 for low-light")
    
    print("\\n🔧 Troubleshooting:")
    print("   • If models don't download: check internet connection")
    print("   • If CUDA errors: update PyTorch and CUDA drivers")
    print("   • If performance is slow: try smaller model (nano)")
    print("   • If detection accuracy is low: try medium model")

def main():
    """Run all tests"""
    print("🚁 Underground Drone System - YOLOv11 Setup Test")
    print("="*60)
    
    tests = [
        ("Model Download & Loading", test_model_download),
        ("Optimized Detector", test_optimized_detector), 
        ("System Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\\n🧪 Running: {test_name}")
        print("-" * 40)
        results[test_name] = test_func()
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your YOLOv11 setup is ready!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    display_recommendations()

if __name__ == "__main__":
    main()
'''
    
    with open("quick_test.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created quick_test.py for testing setup")

def main():
    """Main function to update all existing code"""
    print("🔄 Updating existing code for YOLOv11 compatibility...")
    
    update_human_detector()
    update_main_py()
    create_quick_test_script()
    
    print("\n✅ Code updates complete!")
    print("\nFiles created:")
    print("  • tello_yolo_detection_updated.py - Enhanced detector with auto-download")
    print("  • main_py_updates.py - Updates for your main.py file")
    print("  • quick_test.py - Test script to verify everything works")

if __name__ == "__main__":
    main()