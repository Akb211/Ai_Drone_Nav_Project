
import unittest
import numpy as np
from collision import SLAMSystem
from rl import DQNAgent
class TestDroneSystem(unittest.TestCase):
    def test_state_vector_consistency(self):
        """Test RL state vector has consistent size"""
        agent = DQNAgent(18, 6)
        
        # Test with empty inputs
        state = agent.get_state_from_environment([], {}, {})
        self.assertEqual(len(state), 18)
        
        # Test with human detection
        human_boxes = [[100, 100, 50, 80]]
        slam_data = {"position": [0, 0, 100], "orientation": [0, 0, 0], "safe_directions": ["forward"]}
        telemetry = {"battery": 85}
        
        state = agent.get_state_from_environment(human_boxes, slam_data, telemetry)
        self.assertEqual(len(state), 18)
        self.assertTrue(all(-1 <= x <= 1 for x in state[:6]))  # Position/orientation normalized

    def test_slam_memory_management(self):
        """Test SLAM doesn't consume excessive memory"""
        slam = SLAMSystem()
        initial_memory = len(slam.point_cloud.points)
        
        # Simulate adding many points
        for _ in range(1000):
            slam._update_occupancy_grid(slam.point_cloud)
            
        # Memory should be bounded
        self.assertLess(len(slam.point_cloud.points), 60000)