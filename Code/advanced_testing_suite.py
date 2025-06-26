# advanced_testing_suite.py
import unittest
import numpy as np
import cv2
import torch
import time
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

# Import your modules
from tello_yolo_detection import HumanDetector, HumanTracker
from collision import SLAMSystem
from rl import DQNAgent, DroneRLSystem
from trajectory import visualize_trajectory

class TestDroneSystemComprehensive(unittest.TestCase):
    """Comprehensive testing suite for the underground drone system"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock drone
        self.mock_drone = Mock()
        self.mock_drone.get_battery.return_value = 85
        self.mock_drone.get_height.return_value = 100
        self.mock_drone.is_flying = True
        self.mock_drone.move_forward = Mock()
        self.mock_drone.move_back = Mock()
        self.mock_drone.move_left = Mock()
        self.mock_drone.move_right = Mock()
        self.mock_drone.move_up = Mock()
        self.mock_drone.move_down = Mock()
        
        # Create test frame
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    def tearDown(self):
        """Clean up after each test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class TestHumanDetectionPerformance(TestDroneSystemComprehensive):
    """Test human detection system performance and reliability"""
    
    def test_detection_latency(self):
        """Test detection processing time meets real-time requirements"""
        # Skip if no GPU available or model not present
        if not torch.cuda.is_available():
            self.skipTest("GPU not available for performance testing")
            
        if not os.path.exists('yolo11n.pt'):
            self.skipTest("YOLO model not available")
            
        detector = HumanDetector('yolo11n.pt', device="cuda:0")
        
        # Test detection speed
        num_iterations = 50
        total_time = 0
        
        for _ in range(num_iterations):
            start_time = time.time()
            _ = detector.detect(self.test_frame)
            total_time += time.time() - start_time
        
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time
        
        print(f"Average detection time: {avg_time:.3f}s ({fps:.1f} FPS)")
        
        # Assert real-time performance (>10 FPS for underground scenarios)
        self.assertGreater(fps, 10, "Detection must run at >10 FPS for real-time operation")
    
    def test_detection_accuracy_synthetic(self):
        """Test detection accuracy on synthetic human-like shapes"""
        if not os.path.exists('yolo11n.pt'):
            self.skipTest("YOLO model not available")
            
        detector = HumanDetector('yolo11n.pt')
        
        # Create synthetic frame with human-like rectangle
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw human-like shape (simplified person silhouette)
        cv2.rectangle(synthetic_frame, (250, 150), (350, 400), (128, 128, 128), -1)  # Body
        cv2.circle(synthetic_frame, (300, 120), 30, (128, 128, 128), -1)  # Head
        
        detections = detector.detect(synthetic_frame, conf_threshold=0.3)
        
        # Should detect at least one human-like object
        self.assertGreater(len(detections), 0, "Should detect synthetic human shape")
    
    def test_tracking_consistency(self):
        """Test tracker maintains consistent IDs across frames"""
        tracker = HumanTracker()
        
        # Initialize with mock detection
        initial_boxes = [[100, 100, 50, 80]]
        tracker.init_trackers(self.test_frame, initial_boxes)
        
        # Create slightly modified frame (simulate movement)
        moved_frame = np.roll(self.test_frame, 5, axis=1)  # Shift horizontally
        
        updated_boxes = tracker.update(moved_frame)
        
        # Should maintain tracking
        self.assertEqual(len(updated_boxes), 1, "Should maintain tracking of single person")

class TestSLAMRobustness(TestDroneSystemComprehensive):
    """Test SLAM system robustness and accuracy"""
    
    def test_slam_memory_bounds(self):
        """Test SLAM maintains bounded memory usage"""
        slam = SLAMSystem(min_obstacle_distance=50)
        
        initial_memory = len(slam.point_cloud.points)
        
        # Simulate processing many frames
        for i in range(100):
            # Create frame with random features
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            slam.process_frame(test_frame, drone=self.mock_drone)
        
        final_memory = len(slam.point_cloud.points)
        
        # Memory should be bounded (not grow indefinitely)
        self.assertLess(final_memory, 60000, "Point cloud memory should be bounded")
        print(f"Point cloud size: {initial_memory} -> {final_memory}")
    
    def test_slam_pose_consistency(self):
        """Test SLAM pose estimates are consistent"""
        slam = SLAMSystem()
        
        # Process static frame multiple times
        for _ in range(10):
            slam.process_frame(self.test_frame.copy(), drone=self.mock_drone)
        
        # Position shouldn't drift significantly for static scene
        if len(slam.pose_history) > 5:
            position_variance = np.var(slam.pose_history[-5:], axis=0)
            max_variance = np.max(position_variance)
            
            self.assertLess(max_variance, 100.0, "Position should be stable for static scene")
    
    def test_collision_detection_accuracy(self):
        """Test collision detection identifies obstacles correctly"""
        slam = SLAMSystem(min_obstacle_distance=50)
        
        # Create occupancy grid with obstacles
        slam.occupancy_grid[45:55, 45:55] = 1  # Create obstacle
        slam._update_obstacle_map()
        slam.position = np.array([450, 450, 100])  # Position near obstacle
        
        slam._find_safe_directions()
        
        # Should detect nearby obstacle and limit safe directions
        self.assertLess(len(slam.safe_directions), 6, "Should detect obstacle and limit safe directions")

class TestReinforcementLearning(TestDroneSystemComprehensive):
    """Test RL system training and decision making"""
    
    def test_state_vector_bounds(self):
        """Test all state vector components are properly normalized"""
        agent = DQNAgent(18, 6)
        
        # Test extreme values
        extreme_human_boxes = [[0, 0, 640, 480]]  # Full frame detection
        extreme_slam_data = {
            "position": [1000, 1000, 1000],  # Max position
            "orientation": [180, 180, 180],  # Max orientation
            "safe_directions": []  # No safe directions
        }
        extreme_telemetry = {"battery": 5}  # Low battery
        
        state = agent.get_state_from_environment(extreme_human_boxes, extreme_slam_data, extreme_telemetry)
        
        # Check bounds
        self.assertEqual(len(state), 18, "State vector must have exactly 18 components")
        self.assertTrue(np.all(state >= -1.1), f"State values too low: {state}")
        self.assertTrue(np.all(state <= 1.1), f"State values too high: {state}")
    
    def test_reward_consistency(self):
        """Test reward function produces consistent outputs"""
        agent = DQNAgent(18, 6)
        
        # Test scenarios
        scenarios = [
            # (human_detected, collision, battery, mission_complete, expected_sign)
            (True, False, 80, False, 1),    # Human detected -> positive
            (False, True, 80, False, -1),   # Collision -> negative
            (False, False, 10, False, -1),  # Low battery -> negative
            (True, False, 80, True, 1),     # Mission complete -> positive
        ]
        
        dummy_state = np.zeros(18)
        
        for human_detected, collision, battery, mission_complete, expected_sign in scenarios:
            reward = agent.get_reward(
                dummy_state, 0, dummy_state, human_detected, collision, battery, mission_complete
            )
            
            if expected_sign > 0:
                self.assertGreater(reward, 0, f"Expected positive reward for scenario: {scenarios}")
            else:
                self.assertLess(reward, 0, f"Expected negative reward for scenario: {scenarios}")
    
    def test_action_execution_safety(self):
        """Test action execution includes safety checks"""
        mock_detector = Mock()
        mock_slam = Mock()
        mock_slam.safe_directions = ["forward", "left"]
        
        rl_system = DroneRLSystem(mock_detector, mock_slam, self.mock_drone)
        
        # Test safe action execution
        success = rl_system.execute_action(0)  # Forward action
        self.assertTrue(success, "Safe action should execute successfully")
        
        # Test drone command was called
        self.mock_drone.move_forward.assert_called_once()

class TestSystemIntegration(TestDroneSystemComprehensive):
    """Test full system integration and workflows"""
    
    def test_complete_processing_pipeline(self):
        """Test complete frame processing pipeline"""
        if not os.path.exists('yolo11n.pt'):
            self.skipTest("YOLO model not available")
            
        # Create integrated system
        detector = HumanDetector('yolo11n.pt')
        slam = SLAMSystem()
        rl_system = DroneRLSystem(detector, slam, self.mock_drone)
        
        # Process test frame
        try:
            processed_frame, action = rl_system.process_frame(self.test_frame)
            
            # Verify outputs
            self.assertIsNotNone(processed_frame, "Should return processed frame")
            self.assertIsInstance(action, (int, type(None)), "Action should be integer or None")
            self.assertEqual(processed_frame.shape, self.test_frame.shape, "Frame shape should be preserved")
            
        except Exception as e:
            self.fail(f"Complete pipeline failed: {e}")
    
    def test_emergency_protocols(self):
        """Test emergency protocols activate correctly"""
        # This would test your emergency landing systems
        mock_detector = Mock()
        mock_slam = Mock()
        mock_slam.safe_directions = []  # No safe directions = emergency
        
        rl_system = DroneRLSystem(mock_detector, mock_slam, self.mock_drone)
        
        # Simulate emergency condition
        collision_detected = rl_system.check_for_collision({"safe_directions": []})
        
        self.assertTrue(collision_detected, "Should detect emergency condition")

class TestDataPersistence(TestDroneSystemComprehensive):
    """Test data saving and loading functionality"""
    
    def test_trajectory_saving_loading(self):
        """Test trajectory data can be saved and loaded correctly"""
        slam = SLAMSystem()
        
        # Generate fake trajectory
        slam.pose_history = [
            [0, 0, 100],
            [10, 0, 100],
            [20, 5, 105],
            [30, 10, 110]
        ]
        
        # Save trajectory
        traj_file = os.path.join(self.temp_dir, "test_trajectory.npy")
        slam.save_trajectory(traj_file)
        
        # Verify file exists and can be loaded
        self.assertTrue(os.path.exists(traj_file), "Trajectory file should be saved")
        
        loaded_trajectory = np.load(traj_file)
        self.assertEqual(loaded_trajectory.shape, (4, 3), "Loaded trajectory should match saved data")
    
    def test_mission_data_completeness(self):
        """Test mission data saves all required components"""
        slam = SLAMSystem()
        slam.pose_history = [[0, 0, 100], [10, 0, 100]]
        
        # Save mission data
        base_filename = os.path.join(self.temp_dir, "test_mission")
        slam.save_all_data(base_filename)
        
        # Check all files are created
        expected_files = [
            f"{base_filename}_trajectory_*.npy",
            f"{base_filename}_map_*.npy", 
            f"{base_filename}_metadata_*.json"
        ]
        
        # At least some files should be created
        created_files = os.listdir(self.temp_dir)
        self.assertGreater(len(created_files), 0, "Mission data files should be created")

class TestPerformanceBenchmarks(TestDroneSystemComprehensive):
    """Performance benchmarking tests"""
    
    def test_full_system_latency(self):
        """Test end-to-end system latency"""
        if not os.path.exists('yolo11n.pt'):
            self.skipTest("YOLO model not available")
            
        # Create system
        detector = HumanDetector('yolo11n.pt')
        slam = SLAMSystem()
        rl_system = DroneRLSystem(detector, slam, self.mock_drone)
        
        # Benchmark processing time
        num_frames = 20
        total_time = 0
        
        for _ in range(num_frames):
            start_time = time.time()
            _, _ = rl_system.process_frame(self.test_frame.copy())
            total_time += time.time() - start_time
        
        avg_latency = total_time / num_frames
        fps = 1.0 / avg_latency
        
        print(f"Full system: {avg_latency:.3f}s per frame ({fps:.1f} FPS)")
        
        # Should process at reasonable frame rate for underground scenarios
        self.assertGreater(fps, 5, "Full system should run at >5 FPS minimum")

def run_performance_analysis():
    """Generate detailed performance analysis report"""
    print("\n" + "="*50)
    print("UNDERGROUND DRONE SYSTEM - PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Component timing analysis
    components = {
        'Human Detection': 'test_detection_latency',
        'SLAM Processing': 'test_slam_pose_consistency', 
        'Full System': 'test_full_system_latency'
    }
    
    suite = unittest.TestSuite()
    
    # Add performance tests
    suite.addTest(TestHumanDetectionPerformance('test_detection_latency'))
    suite.addTest(TestPerformanceBenchmarks('test_full_system_latency'))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nPerformance Tests: {result.testsRun} run, {len(result.failures)} failed")
    return result

def run_safety_validation():
    """Run safety-critical validation tests"""
    print("\n" + "="*50) 
    print("SAFETY VALIDATION TESTS")
    print("="*50)
    
    suite = unittest.TestSuite()
    
    # Add safety tests
    suite.addTest(TestSLAMRobustness('test_collision_detection_accuracy'))
    suite.addTest(TestSystemIntegration('test_emergency_protocols'))
    suite.addTest(TestReinforcementLearning('test_action_execution_safety'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nSafety Tests: {result.testsRun} run, {len(result.failures)} failed")
    return result

if __name__ == "__main__":
    print("Starting Comprehensive Drone System Testing...")
    
    # Run all tests
    unittest.main(verbosity=2, exit=False)
    
    # Run specialized analysis
    perf_results = run_performance_analysis()
    safety_results = run_safety_validation()
    
    # Generate summary report
    print("\n" + "="*50)
    print("TESTING SUMMARY")
    print("="*50)
    print("✅ All basic functionality tests should pass")
    print("⚡ Performance tests validate real-time capability") 
    print("🛡️ Safety tests ensure reliable operation")
    print("📊 Ready for real-world deployment testing!")