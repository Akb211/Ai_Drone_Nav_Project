   
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from djitellopy import Tello
import time

class SLAMSystem:
    def __init__(self, min_obstacle_distance=50):
        # Map representation
        self.point_cloud = o3d.geometry.PointCloud()
        self.obstacle_map = None
        self.map_resolution = 10  # cm per cell
        self.map_size = (1000, 1000)  # 10m x 10m default map
        
        # Initialize empty occupancy grid
        self.occupancy_grid = np.zeros((
            self.map_size[0] // self.map_resolution, 
            self.map_size[1] // self.map_resolution
        ), dtype=np.int8)
        
        # Drone state estimation
        self.position = np.zeros(3)  # [x, y, z] in cm
        self.orientation = np.zeros(3)  # [roll, pitch, yaw] in degrees
        self.pose_history = []
        self.home_position = np.zeros(3)  # Starting position
        self.has_set_home = False
        
        # Feature detection and tracking for visual odometry
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None
        
        # Collision avoidance parameters
        self.min_obstacle_distance = min_obstacle_distance
        self.safe_directions = []
        
        # SLAM parameters
        self.keyframes = []
        self.loop_closure_threshold = 0.75
    def get_path_to_home(self):
        # Calculate direct distance to home
        distance = np.linalg.norm(self.position - self.home_position)
        
        # Use path planning to find a safe path home
        path = self.plan_path(self.home_position)
        
        if not path:
            # If no path found, create a simple direct path (may not avoid obstacles)
            # In a real system, you'd want a more sophisticated fallback
            print("Warning: Could not find safe path home. Using direct path.")
            path = [self.position.copy(), self.home_position.copy()]
        
        return path, distance
    def process_frame(self, frame, depth_frame=None, drone=None):
        # Set home position if not set yet
        if not self.has_set_home:
            self.home_position = self.position.copy()
            self.has_set_home = True
            print(f"Home position set to: {self.home_position}")
            
        # Convert frame to grayscale for feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # Update position using visual odometry
        if self.prev_kp is not None and self.prev_des is not None and des is not None:
            self._update_position_from_features(kp, des)
            
        # If we have drone telemetry, use it to refine position estimate
        if drone is not None:
            self._update_position_from_sensors(drone)
            
        # Process depth information (either from depth frame or estimate from disparity)
        if depth_frame is not None:
            self._update_map_from_depth(depth_frame, kp)
        else:
            # If no depth frame, we can estimate some basic depth from stereo or motion
            self._estimate_depth_from_motion(gray, kp)
        
        # Update obstacle map and find safe directions
        self._update_obstacle_map()
        self._find_safe_directions()
        
        # Store current features for next frame
        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = gray
        
        # Draw visualization
        processed_frame = self._visualize_slam(frame, kp)
        
        # Check for loop closures periodically
        self._check_loop_closure()
        
        # Store pose history
        self.pose_history.append(self.position.copy())
        
        return processed_frame, self.safe_directions
    

    def correct_drift_with_imu(self, imu_data):
        """Use IMU data to correct SLAM drift"""
        if imu_data and len(self.pose_history) > 10:
            # Simple drift correction using IMU
            imu_orientation = np.array([imu_data.get('roll', 0), 
                                    imu_data.get('pitch', 0), 
                                    imu_data.get('yaw', 0)])
            
            # Weighted average with SLAM orientation
            alpha = 0.7  # Trust SLAM more than IMU
            self.orientation = alpha * self.orientation + (1 - alpha) * imu_orientation
    def _update_position_from_features(self, kp, des):
            if len(kp) < 10 or des is None or self.prev_des is None:
                return
            
            # Match features
            matches = self.bf_matcher.match(self.prev_des, des)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Take only good matches
            good_matches = matches[:int(len(matches) * 0.7)]
            
            if len(good_matches) < 8:
                return
                
            # Extract matched keypoints
            prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
            curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])
            
            # Calculate essential matrix
            # Use the dimensions of the previous frame that we stored
            h, w = self.prev_frame.shape
            E, mask = cv2.findEssentialMat(prev_pts, curr_pts, 
                                        focal=500, # Approximate focal length
                                        pp=(w//2, h//2), 
                                        method=cv2.RANSAC, 
                                        prob=0.999, 
                                        threshold=1.0)
            
            if E is None:
                return
                
            # Recover relative pose
            _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, 
                                        focal=500, 
                                        pp=(w//2, h//2))
            
            # Scale translation (in a real system, this would come from depth or other sensors)
            scale = 10.0  # Arbitrary scale factor, to be replaced with actual scale estimation
            
            # Update position (simplified - in a real implementation, this would use proper pose composition)
            translation = scale * t.flatten()
            
            # Convert rotation matrix to Euler angles
            r = Rotation.from_matrix(R)
            euler = r.as_euler('xyz', degrees=True)
            
            # Update position and orientation
            # Note: This is simplified - a real system would use proper pose composition
            self.position += translation
            self.orientation += euler
    
    def _update_position_from_sensors(self, drone):
        # Get sensor data from the Tello drone
        try:
            # Note: Tello doesn't provide absolute position, but we can use other sensors
            height = drone.get_height() * 10  # Convert to cm
            
            # For Tello, we don't have direct access to IMU data through get_attitude()
            # Instead, we'll use simpler functions available in the API
            
            # Update z position from barometer
            if height > 0:
                self.position[2] = height
                
        except Exception as e:
            # Handle potential errors when getting sensor data
            print(f"Warning: Could not get all telemetry data: {e}")
            pass
    
    def _update_map_from_depth(self, depth_frame, keypoints):
        # In a real implementation, this would project 2D points to 3D using the depth frame
        # For now, we'll use a simplified version
        
        # Create new point cloud from current frame
        new_points = []
        
        # For each keypoint, get 3D position using depth
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check if within depth frame bounds
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                depth = depth_frame[y, x]
                
                # Skip invalid depth values
                if depth <= 0:
                    continue
                
                # Convert image coordinates to 3D world coordinates (simplified)
                # In a real system, this would use camera intrinsics
                fx, fy = 500, 500  # Approximate focal length
                cx, cy = depth_frame.shape[1]//2, depth_frame.shape[0]//2  # Principal point
                
                # Calculate 3D point
                x_world = (x - cx) * depth / fx
                y_world = (y - cy) * depth / fy
                z_world = depth
                
                new_points.append([x_world, y_world, z_world])
        
        if len(new_points) > 0:
            new_cloud = o3d.geometry.PointCloud()
            new_cloud.points = o3d.utility.Vector3dVector(np.array(new_points))
            
            # Transform to world coordinates
            R = Rotation.from_euler('xyz', self.orientation, degrees=True).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = self.position
            new_cloud.transform(T)
            
            # MEMORY MANAGEMENT: Limit point cloud size
            self.point_cloud += new_cloud
            if len(self.point_cloud.points) > 50000:  # Limit to 50k points
                # Downsample using voxel grid
                self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size=5.0)
            
            self._update_occupancy_grid(new_cloud)
    
    def _estimate_depth_from_motion(self, gray_frame, keypoints):
        if self.prev_frame is None or len(keypoints) == 0 or self.prev_kp is None or len(self.prev_kp) == 0:
            return
            
        # Convert keypoints to numpy arrays - add error checking here
        try:
            prev_pts = np.float32([kp.pt for kp in self.prev_kp]).reshape(-1, 1, 2)
            
            # Calculate optical flow with better error handling
            try:
                flow_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray_frame, prev_pts, None
                )
                
                # Filter only valid points
                valid_indices = status.ravel() == 1
                
                # Check if we have any valid points
                if np.sum(valid_indices) == 0:
                    return
                    
                good_prev = prev_pts[valid_indices].reshape(-1, 2)
                good_curr = flow_pts[valid_indices].reshape(-1, 2)
                
                if len(good_prev) > 0:
                    # Calculate flow magnitude safely
                    flow = good_curr - good_prev
                    flow_magnitude = np.sqrt(np.sum(flow**2, axis=1))
                    
                    # Rest of your function...
                    
            except Exception as e:
                print(f"Warning: Error in optical flow calculation: {e}")
                return
                
        except Exception as e:
            print(f"Warning: Error processing keypoints: {e}")
            return
    
    def _update_occupancy_grid(self, point_cloud):
        # Convert point cloud to numpy array
        points = np.asarray(point_cloud.points)
        
        if len(points) == 0:
            return
            
        # Transform points to grid coordinates
        grid_points = np.floor((points[:, :2] + self.map_size[0]/2) / self.map_resolution).astype(int)
        
        # Filter points within grid bounds
        valid_idx = (
            (grid_points[:, 0] >= 0) & 
            (grid_points[:, 0] < self.occupancy_grid.shape[0]) &
            (grid_points[:, 1] >= 0) & 
            (grid_points[:, 1] < self.occupancy_grid.shape[1])
        )
        
        grid_points = grid_points[valid_idx]
        
        # Update occupancy grid
        for point in grid_points:
            self.occupancy_grid[point[0], point[1]] = 1
    
    def _update_obstacle_map(self):
        # Dilate occupancy grid to create safety margin around obstacles
        kernel = np.ones((3, 3), np.uint8)
        self.obstacle_map = cv2.dilate(self.occupancy_grid.astype(np.uint8), kernel)
    
    def _find_safe_directions(self):
        # Get current position in grid coordinates
        curr_x, curr_y = int((self.position[0] + self.map_size[0]/2) / self.map_resolution), \
                         int((self.position[1] + self.map_size[1]/2) / self.map_resolution)
        
        # Check if current position is within map bounds
        if (0 <= curr_x < self.obstacle_map.shape[0] and 
            0 <= curr_y < self.obstacle_map.shape[1]):
            
            # Check directions around current position
            directions = [
                ("forward", 0, 1), 
                ("backward", 0, -1),
                ("left", -1, 0),
                ("right", 1, 0),
                ("up", 0, 0),  # Z-axis not represented in 2D grid
                ("down", 0, 0)  # Z-axis not represented in 2D grid
            ]
            
            self.safe_directions = []
            
            for direction, dx, dy in directions:
                # Check if direction is safe (no obstacles within min_obstacle_distance)
                is_safe = True
                
                # Special handling for up and down
                if direction in ["up", "down"]:
                    # For now, we'll assume up/down are safe if no direct obstacles
                    # In a real system, this would check ceiling and floor
                    self.safe_directions.append(direction)
                    continue
                
                # Check multiple steps in the direction
                steps = int(self.min_obstacle_distance / self.map_resolution)
                for step in range(1, steps + 1):
                    check_x, check_y = curr_x + dx * step, curr_y + dy * step
                    
                    # Check if position is within map bounds
                    if not (0 <= check_x < self.obstacle_map.shape[0] and 
                            0 <= check_y < self.obstacle_map.shape[1]):
                        is_safe = False
                        break
                    
                    # Check if position is obstacle-free
                    if self.obstacle_map[check_x, check_y] > 0:
                        is_safe = False
                        break
                
                if is_safe:
                    self.safe_directions.append(direction)
    
    def _check_loop_closure(self):
        # In a real implementation, this would detect revisited locations
        # and update the map to correct drift
        # Simplified version: just store keyframes periodically
        
        # Add current frame as keyframe every 20 frames (arbitrary)
        if len(self.pose_history) % 20 == 0 and self.prev_frame is not None:
            self.keyframes.append({
                'frame': self.prev_frame.copy(),
                'position': self.position.copy(),
                'keypoints': self.prev_kp,
                'descriptors': self.prev_des
            })
            
            # Check for loop closures with previous keyframes
            if len(self.keyframes) > 5:  # Need some history
                self._detect_loop_closures()
    
    def _detect_loop_closures(self):
        # This is a simplified version of loop closure detection
        # In a real implementation, this would be more sophisticated
        
        # Get the latest keyframe
        current = self.keyframes[-1]
        current_des = current['descriptors']
        
        # Compare with all previous keyframes
        for i, keyframe in enumerate(self.keyframes[:-1]):
            if keyframe['descriptors'] is None or current_des is None:
                continue
                
            # Match descriptors
            matches = self.bf_matcher.match(keyframe['descriptors'], current_des)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate match quality
            if len(matches) > 20:
                # Calculate average distance of top matches
                avg_distance = sum(m.distance for m in matches[:20]) / 20
                match_quality = 1.0 - min(avg_distance / 100.0, 1.0)  # Normalize
                
                # If match quality is high enough, we found a loop closure
                if match_quality > self.loop_closure_threshold:
                    print(f"Loop closure detected with keyframe {i} (quality: {match_quality:.2f})")
                    
                    # In a real implementation, this would update the map and correct positions
                    # For now, just print the detection
    
    def _visualize_slam(self, frame, keypoints):

        # Draw keypoints
        vis_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
        
        # Draw position and orientation
        pos_text = f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f})"
        cv2.putText(vis_frame, pos_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw distance to home
        home_dist = np.linalg.norm(self.position - self.home_position)
        home_text = f"Home: {home_dist:.1f} cm"
        cv2.putText(vis_frame, home_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw safe directions
        dir_text = f"Safe: {', '.join(self.safe_directions)}"
        cv2.putText(vis_frame, dir_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw mini-map (top-down view)
        map_size = 150
        mini_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        
        # Draw obstacles from occupancy grid
        if self.obstacle_map is not None:
            # Calculate region to show in mini-map (centered on drone)
            center_x = int((self.position[0] + self.map_size[0]/2) / self.map_resolution)
            center_y = int((self.position[1] + self.map_size[1]/2) / self.map_resolution)
            
            half_size = map_size // 2
            min_x, max_x = center_x - half_size, center_x + half_size
            min_y, max_y = center_y - half_size, center_y + half_size
            
            # Ensure bounds are within map
            min_x = max(0, min(min_x, self.obstacle_map.shape[0] - map_size))
            min_y = max(0, min(min_y, self.obstacle_map.shape[1] - map_size))
            max_x = min(self.obstacle_map.shape[0], max_x)
            max_y = min(self.obstacle_map.shape[1], max_y)
            
            # Draw visible portion of map
            for i in range(min_x, min(max_x, min_x + map_size)):
                for j in range(min_y, min(max_y, min_y + map_size)):
                    if 0 <= i < self.obstacle_map.shape[0] and 0 <= j < self.obstacle_map.shape[1]:
                        if self.obstacle_map[i, j] > 0:
                            # Draw obstacle
                            map_i, map_j = i - min_x, j - min_y
                            if 0 <= map_i < map_size and 0 <= map_j < map_size:
                                mini_map[map_i, map_j] = [0, 0, 255]  # Red for obstacles
        
        # Draw trajectory
        if len(self.pose_history) > 1:
            for i in range(1, len(self.pose_history)):
                p1 = self.pose_history[i-1]
                p2 = self.pose_history[i]
                
                # Convert to mini-map coordinates
                p1_x = int((p1[0] + self.map_size[0]/2) / self.map_resolution) - min_x
                p1_y = int((p1[1] + self.map_size[1]/2) / self.map_resolution) - min_y
                p2_x = int((p2[0] + self.map_size[0]/2) / self.map_resolution) - min_x
                p2_y = int((p2[1] + self.map_size[1]/2) / self.map_resolution) - min_y
                
                # Draw line if within mini-map bounds
                if (0 <= p1_x < map_size and 0 <= p1_y < map_size and
                    0 <= p2_x < map_size and 0 <= p2_y < map_size):
                    cv2.line(mini_map, (p1_y, p1_x), (p2_y, p2_x), (0, 255, 0), 1)
        
        # Draw current position
        center_on_map_x = int(map_size / 2)
        center_on_map_y = int(map_size / 2)
        cv2.circle(mini_map, (center_on_map_y, center_on_map_x), 5, (255, 255, 0), -1)
        
        # Draw home position
        home_x = int((self.home_position[0] + self.map_size[0]/2) / self.map_resolution) - min_x
        home_y = int((self.home_position[1] + self.map_size[1]/2) / self.map_resolution) - min_y
        if 0 <= home_x < map_size and 0 <= home_y < map_size:
            cv2.circle(mini_map, (home_y, home_x), 7, (255, 0, 255), -1)  # Purple for home
        
        # Draw orientation
        yaw_rad = np.radians(self.orientation[2])
        end_x = center_on_map_x + int(15 * np.sin(yaw_rad))
        end_y = center_on_map_y + int(15 * np.cos(yaw_rad))
        cv2.line(mini_map, (center_on_map_y, center_on_map_x), (end_y, end_x), (255, 255, 0), 2)
        
        # Place mini-map on the main visualization
        vis_frame[10:10+map_size, vis_frame.shape[1]-map_size-10:vis_frame.shape[1]-10] = mini_map
        
        return vis_frame

    def get_map(self):

        return self.occupancy_grid, self.point_cloud
    
    def plan_path(self, target_position):

        # Convert positions to grid coordinates
        start_x, start_y = int((self.position[0] + self.map_size[0]/2) / self.map_resolution), \
                          int((self.position[1] + self.map_size[1]/2) / self.map_resolution)
        
        target_x, target_y = int((target_position[0] + self.map_size[0]/2) / self.map_resolution), \
                           int((target_position[1] + self.map_size[1]/2) / self.map_resolution)
        
        # Check if positions are within map bounds
        if not (0 <= start_x < self.obstacle_map.shape[0] and 
                0 <= start_y < self.obstacle_map.shape[1] and
                0 <= target_x < self.obstacle_map.shape[0] and
                0 <= target_y < self.obstacle_map.shape[1]):
            print("Target position is outside the map bounds")
            return []
        
        # Check if target position is obstacle-free
        if self.obstacle_map[target_x, target_y] > 0:
            print("Target position is blocked by an obstacle")
            return []
        
        # Simple A* path planning
        # This is a simplified implementation - a real one would be more optimized
        
        # Priority queue for A*
        from queue import PriorityQueue
        open_set = PriorityQueue()
        open_set.put((0, (start_x, start_y)))
        
        # Tracking dictionaries
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self._heuristic((start_x, start_y), (target_x, target_y))}
        
        # Track visited nodes
        open_set_hash = {(start_x, start_y)}
        
        while not open_set.empty():
            # Get node with lowest f_score
            _, current = open_set.get()
            open_set_hash.remove(current)
            
            # Check if we've reached the goal
            if current == (target_x, target_y):
                # Reconstruct path
                path = []
                while current in came_from:
                    # Convert grid coordinates back to world coordinates
                    x = (current[0] * self.map_resolution) - self.map_size[0]/2
                    y = (current[1] * self.map_resolution) - self.map_size[1]/2
                    path.append([x, y, self.position[2]])  # Keep same height for simplicity
                    current = came_from[current]
                
                # Add start position
                x = (start_x * self.map_resolution) - self.map_size[0]/2
                y = (start_y * self.map_resolution) - self.map_size[1]/2
                path.append([x, y, self.position[2]])
                
                # Reverse to get path from start to goal
                path.reverse()
                return path
            
            # Check neighbors
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor_x, neighbor_y = current[0] + dx, current[1] + dy
                
                # Check if neighbor is within bounds
                if not (0 <= neighbor_x < self.obstacle_map.shape[0] and 
                        0 <= neighbor_y < self.obstacle_map.shape[1]):
                    continue
                
                # Check if neighbor is obstacle-free
                if self.obstacle_map[neighbor_x, neighbor_y] > 0:
                    continue
                
                neighbors.append((neighbor_x, neighbor_y))
            
            for neighbor in neighbors:
                # Calculate tentative g_score
                # Diagonal movements cost more
                if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2:
                    tentative_g_score = g_score[current] + 1.414  # sqrt(2)
                else:
                    tentative_g_score = g_score[current] + 1
                
                # Check if this path is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, (target_x, target_y))
                    
                    if neighbor not in open_set_hash:
                        open_set.put((f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        print("No path found")
        return []
    
    def _heuristic(self, a, b):

        # Euclidean distance
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    def save_trajectory(self, filename='trajectory.npy'):

      if len(self.pose_history) > 0:
          # Convert pose history to numpy array
          trajectory = np.array(self.pose_history)
        
          # Save trajectory to file
          np.save(filename, trajectory)
          print(f"Trajectory saved to {filename}")
      else:
          print("No trajectory data to save")

    def save_map(self, filename='occupancy_map.npy'):

        if self.occupancy_grid is not None:
            # Save occupancy grid to file
            np.save(filename, self.occupancy_grid)
            print(f"Occupancy map saved to {filename}")
        else:
            print("No map data to save")

    def save_point_cloud(self, filename='point_cloud.ply'):

        if self.point_cloud and len(self.point_cloud.points) > 0:
            # Save point cloud to PLY file
            o3d.io.write_point_cloud(filename, self.point_cloud)
            print(f"Point cloud saved to {filename}")
        else:
            print("No point cloud data to save")

    def save_all_data(self, base_filename='mission_data'):

        # Generate timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save trajectory
        traj_filename = f"{base_filename}_trajectory_{timestamp}.npy"
        self.save_trajectory(traj_filename)
        
        # Save occupancy map
        map_filename = f"{base_filename}_map_{timestamp}.npy"
        self.save_map(map_filename)
        
        # Save point cloud
        pc_filename = f"{base_filename}_pointcloud_{timestamp}.ply"
        self.save_point_cloud(pc_filename)
        
        # Save metadata (e.g., map resolution, size)
        metadata = {
            'timestamp': timestamp,
            'map_resolution': self.map_resolution,
            'map_size': self.map_size,
            'home_position': self.home_position.tolist(),
            'trajectory_length': len(self.pose_history),
            'trajectory_file': traj_filename,
            'map_file': map_filename,
            'pointcloud_file': pc_filename
        }
        
        import json
        with open(f"{base_filename}_metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"All mission data saved with timestamp {timestamp}")
    
    def visualize_3d_map(self, window_name="3D Map"):
        # Only visualize if we have enough points
        if len(self.point_cloud.points) < 10:
            return None
            
        # Create a copy of the point cloud for visualization
        vis_cloud = o3d.geometry.PointCloud()
        vis_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.point_cloud.points))
        vis.get_render_option().point_size = 5.0  # Increase point size for visibility

        # Add colors to the point cloud (e.g., height-based coloring)
        points = np.asarray(vis_cloud.points)
        colors = np.zeros_like(points)
        # Color based on height (z-coordinate)
        min_z = np.min(points[:, 2]) if len(points) > 0 else 0
        max_z = np.max(points[:, 2]) if len(points) > 0 else 1
        if max_z > min_z:
            for i in range(len(points)):
                norm_z = (points[i, 2] - min_z) / (max_z - min_z)
                colors[i] = [norm_z, 0.5, 1 - norm_z]  # Blue to purple to red
        
        vis_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Create a simple image of the point cloud for display in OpenCV window
        # We'll use Open3D's built-in visualization to create a snapshot
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=400, height=400, visible=False)
        vis.add_geometry(vis_cloud)
        
        # Set up reasonable camera viewpoint
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # Update geometry and render
        vis.update_geometry(vis_cloud)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image from visualizer
        map_image = np.asarray(vis.capture_screen_float_buffer())
        map_image = (map_image * 255).astype(np.uint8)
        
        # Clean up visualizer
        vis.destroy_window()
        
        return map_image
    
    def visualize_2d_map(self, window_size=(800, 800)):
        if self.obstacle_map is None:
            return None
        
        # Create a higher contrast visualization
        map_image = np.ones((self.obstacle_map.shape[0], self.obstacle_map.shape[1], 3), dtype=np.uint8) * 255  # White background
        
        # Fill free space with white (already set as background)
        
        # Fill unexplored space with light gray
        unexplored = (self.occupancy_grid == 0) & (np.sum(self.pose_history, axis=0) == 0)
        map_image[unexplored] = [200, 200, 200]
        
        # Fill obstacles with black
        obstacles = self.obstacle_map > 0
        map_image[obstacles] = [50, 50, 50]  # Dark gray/black for obstacles
        
        # Mark the drone's position and orientation
        pos_x = int((self.position[0] + self.map_size[0]/2) / self.map_resolution)
        pos_y = int((self.position[1] + self.map_size[1]/2) / self.map_resolution)
        
        # Draw trajectory with better visibility
        if len(self.pose_history) > 1:
            for i in range(1, len(self.pose_history)):
                p1 = self.pose_history[i-1]
                p2 = self.pose_history[i]
                
                # Convert to map coordinates
                p1_x = int((p1[0] + self.map_size[0]/2) / self.map_resolution)
                p1_y = int((p1[1] + self.map_size[1]/2) / self.map_resolution)
                p2_x = int((p2[0] + self.map_size[0]/2) / self.map_resolution)
                p2_y = int((p2[1] + self.map_size[1]/2) / self.map_resolution)
                
                # Draw line if within map bounds
                if (0 <= p1_x < map_image.shape[0] and 0 <= p1_y < map_image.shape[1] and
                    0 <= p2_x < map_image.shape[0] and 0 <= p2_y < map_image.shape[1]):
                    # Draw thicker blue line for trajectory
                    cv2.line(map_image, (p1_y, p1_x), (p2_y, p2_x), (255, 0, 0), 2)
        
        # Draw current position and orientation more clearly
        if 0 <= pos_x < map_image.shape[0] and 0 <= pos_y < map_image.shape[1]:
            # Draw drone as a red circle
            cv2.circle(map_image, (pos_y, pos_x), 5, (0, 0, 255), -1)
            
            # Draw orientation line
            yaw_rad = np.radians(self.orientation[2])
            end_x = pos_x + int(15 * np.sin(yaw_rad))
            end_y = pos_y + int(15 * np.cos(yaw_rad))
            
            if (0 <= end_x < map_image.shape[0] and 0 <= end_y < map_image.shape[1]):
                cv2.line(map_image, (pos_y, pos_x), (end_y, end_x), (0, 0, 255), 2)
        
        # Add a border and grid lines for better spatial understanding
        # Draw grid lines every 50cm
        grid_interval = int(50 / self.map_resolution)
        for x in range(0, map_image.shape[0], grid_interval):
            cv2.line(map_image, (0, x), (map_image.shape[1]-1, x), (230, 230, 230), 1)
        for y in range(0, map_image.shape[1], grid_interval):
            cv2.line(map_image, (y, 0), (y, map_image.shape[0]-1), (230, 230, 230), 1)
        
        # Add coordinate axes for orientation
        center_x = map_image.shape[0] // 2
        center_y = map_image.shape[1] // 2
        axis_length = 50
        
        # X-axis (red)
        cv2.arrowedLine(map_image, (center_y, center_x), (center_y + axis_length, center_x), (0, 0, 255), 2)
        # Y-axis (green)
        cv2.arrowedLine(map_image, (center_y, center_x), (center_y, center_x - axis_length), (0, 255, 0), 2)
        
        # Add scale indicator
        scale_x = 50
        scale_y = map_image.shape[0] - 30
        cv2.line(map_image, (scale_x, scale_y), (scale_x + grid_interval, scale_y), (0, 0, 0), 2)
        cv2.putText(map_image, "50cm", (scale_x, scale_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Resize to requested window size
        map_image = cv2.resize(map_image, window_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply a subtle blur to smooth pixelation
        map_image = cv2.GaussianBlur(map_image, (3, 3), 0)
        
        return map_image
    
    def save_maps(self, base_filename='mission_maps'):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Generate enhanced 2D map
        raw_map_2d = self.visualize_2d_map((1200, 1200))  # Higher resolution
        
        if raw_map_2d is not None:
            # Apply post-processing for a cleaner look
            # Apply median blur to reduce noise
            enhanced_map = cv2.medianBlur(raw_map_2d, 5)
            
            # Improve contrast
            lab = cv2.cvtColor(enhanced_map, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_map = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Save both versions
            map2d_filename = f"{base_filename}_2D_{timestamp}.png"
            enhanced_filename = f"{base_filename}_2D_enhanced_{timestamp}.png"
            
            cv2.imwrite(map2d_filename, raw_map_2d)
            cv2.imwrite(enhanced_filename, enhanced_map)
            
            print(f"2D maps saved to {map2d_filename} and {enhanced_filename}")
        
        # Save 3D map if we have enough points
        if len(self.point_cloud.points) > 10:
            # Save as PNG visualization
            map_3d = self.visualize_3d_map()
            if map_3d is not None:
                map3d_filename = f"{base_filename}_3D_{timestamp}.png"
                cv2.imwrite(map3d_filename, map_3d)
                print(f"3D map visualization saved to {map3d_filename}")
            
            # Also save the actual point cloud for potential future use
            ply_filename = f"{base_filename}_pointcloud_{timestamp}.ply"
            o3d.io.write_point_cloud(ply_filename, self.point_cloud)
            print(f"Point cloud saved to {ply_filename}")
            
    
    def save_trajectory(self, filename='trajectory.npy'):
        if len(self.pose_history) > 0:
            # Convert pose history to numpy array
            trajectory = np.array(self.pose_history)
            
            # Save trajectory to file
            np.save(filename, trajectory)
            print(f"Trajectory saved to {filename}")
        else:
            print("No trajectory data to save")