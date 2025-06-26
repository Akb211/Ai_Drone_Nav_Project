import cv2
import numpy as np
import time
import argparse
import os
from djitellopy import Tello
from trajectory import visualize_trajectory, load_and_display_mission_data, trajectory_video_generator, compare_trajectories, generate_mission_report
from tello_yolo_detection import HumanDetector, HumanTracker
from collision import SLAMSystem
from rl import DroneRLSystem
import matplotlib.pyplot as plt               
import threading
from queue import Queue

def return_to_home(tello, slam_system, frame_width, frame_height):
    print("Returning to home position...")
    
    # Get path to home
    path, distance = slam_system.get_path_to_home()
    
    if not path or len(path) < 2:
        print("No valid path to home found. Attempting direct return.")
        # Try to move directly towards home
        dx = slam_system.home_position[0] - slam_system.position[0]
        dy = slam_system.home_position[1] - slam_system.position[1]
        dz = slam_system.home_position[2] - slam_system.position[2]
        
        # Move in x direction
        if abs(dx) > 20:
            try:
                if dx > 0:
                    tello.move_forward(min(int(abs(dx)), 100))
                else:
                    tello.move_back(min(int(abs(dx)), 100))
                time.sleep(1)
            except Exception as e:
                print(f"Error moving in x direction: {e}")
        
        # Move in y direction
        if abs(dy) > 20:
            try:
                if dy > 0:
                    tello.move_right(min(int(abs(dy)), 100))
                else:
                    tello.move_left(min(int(abs(dy)), 100))
                time.sleep(1)
            except Exception as e:
                print(f"Error moving in y direction: {e}")
        
        # Move in z direction
        if abs(dz) > 20:
            try:
                if dz > 0:
                    tello.move_up(min(int(abs(dz)), 100))
                else:
                    tello.move_down(min(int(abs(dz)), 100))
                time.sleep(1)
            except Exception as e:
                print(f"Error moving in z direction: {e}")
    else:
        # Follow the path home
        print(f"Following path home with {len(path)} waypoints")
        
        # Create visualization for return path
        path_vis = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Draw the path
        for i in range(1, len(path)):
            # Scale positions to fit in the visualization frame
            start_x = int((path[i-1][0] + 500) / 1000 * frame_width)
            start_y = int((path[i-1][1] + 500) / 1000 * frame_height)
            end_x = int((path[i][0] + 500) / 1000 * frame_width)
            end_y = int((path[i][1] + 500) / 1000 * frame_height)
            
            # Draw line segment
            cv2.line(path_vis, (start_y, start_x), (end_y, end_x), (0, 255, 0), 2)
        
        # Draw home position
        home_x = int((slam_system.home_position[0] + 500) / 1000 * frame_width)
        home_y = int((slam_system.home_position[1] + 500) / 1000 * frame_height)
        cv2.circle(path_vis, (home_y, home_x), 10, (255, 0, 255), -1)
        
        # Show the return path
        cv2.imshow("Return Path", path_vis)
        cv2.waitKey(1000)  # Show for 1 second
        
        # Follow the path by moving to each waypoint
        for i in range(1, min(len(path), 5)):  # Limit to first 5 waypoints to avoid too complex paths
            # Calculate relative movement
            dx = path[i][0] - slam_system.position[0]
            dy = path[i][1] - slam_system.position[1]
            dz = path[i][2] - slam_system.position[2]
            
            print(f"Moving to waypoint {i}: dx={dx:.1f}, dy={dy:.1f}, dz={dz:.1f}")
            
            # Move in x direction
            if abs(dx) > 20:
                try:
                    if dx > 0:
                        tello.move_forward(min(int(abs(dx)), 100))
                    else:
                        tello.move_back(min(int(abs(dx)), 100))
                    time.sleep(1)
                except Exception as e:
                    print(f"Error moving in x direction: {e}")
            
            # Move in y direction
            if abs(dy) > 20:
                try:
                    if dy > 0:
                        tello.move_right(min(int(abs(dy)), 100))
                    else:
                        tello.move_left(min(int(abs(dy)), 100))
                    time.sleep(1)
                except Exception as e:
                    print(f"Error moving in y direction: {e}")
            
            # Move in z direction
            if abs(dz) > 20:
                try:
                    if dz > 0:
                        tello.move_up(min(int(abs(dz)), 100))
                    else:
                        tello.move_down(min(int(abs(dz)), 100))
                    time.sleep(1)
                except Exception as e:
                    print(f"Error moving in z direction: {e}")
            
            # Update current position
    
    # Final approach - move more precisely to home position
    print("Final approach to home position...")
    
    # Get current distance to home
    current_distance = np.linalg.norm(slam_system.position - slam_system.home_position)
    
    if current_distance > 50:  # If we're still far from home
        dx = slam_system.home_position[0] - slam_system.position[0]
        dy = slam_system.home_position[1] - slam_system.position[1]
        dz = slam_system.home_position[2] - slam_system.position[2]
        
        # Make final adjustments
        try:
            if abs(dx) > 20:
                if dx > 0:
                    tello.move_forward(min(int(abs(dx)), 50))
                else:
                    tello.move_back(min(int(abs(dx)), 50))
                time.sleep(1)
            
            if abs(dy) > 20:
                if dy > 0:
                    tello.move_right(min(int(abs(dy)), 50))
                else:
                    tello.move_left(min(int(abs(dy)), 50))
                time.sleep(1)
            
            if abs(dz) > 20:
                if dz > 0:
                    tello.move_up(min(int(abs(dz)), 50))
                else:
                    tello.move_down(min(int(abs(dz)), 50))
                time.sleep(1)    
        except Exception as e:
            print(f"Error moving in final approach: {e}")
            
def parse_args():
    parser = argparse.ArgumentParser(description='Underground Drone System with Human Detection, SLAM, and RL')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Path to YOLOv11 model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run YOLO model on')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--output', type=str, default='drone_mission.avi', help='Output video path')
    parser.add_argument('--rl_model', type=str, default=None, help='Path to RL model to load')
    parser.add_argument('--training', action='store_true', help='Enable RL training mode')
    parser.add_argument('--sim', action='store_true', help='Run in simulation mode (no actual drone)')
    parser.add_argument('--save-data', action='store_true', help='Save mission data (trajectory and map)')
    parser.add_argument('--vis-data', type=str, default=None, help='Visualize saved trajectory file')
    parser.add_argument('--vis-metadata', type=str, default=None, help='Visualize saved mission metadata')
    parser.add_argument('--gen-video', type=str, default=None, help='Generate video from trajectory file')
    parser.add_argument('--compare-trajectories', nargs='+', default=None, help='Compare multiple trajectory files')
    parser.add_argument('--trajectory-labels', nargs='+', default=None, help='Labels for trajectory comparison')
    parser.add_argument('--analysis-dir', type=str, default='mission_analysis', help='Directory for analysis outputs')
    parser.add_argument('--detection-interval', type=int, default=10, 
                  help='Run detection every N frames (higher values improve performance)')
    return parser.parse_args()

def setup_drone():
    tello = Tello()
    try:
        tello.connect()
        print("Tello Battery:", tello.get_battery())
        
        # Set video stream settings for better performance
        tello.set_video_fps(Tello.FPS_15)
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        
        # Start video stream with a delay
        tello.streamon()
        time.sleep(2)  # Wait for the stream to initialize
        
    except Exception as e:
        print(f"Error connecting to Tello: {e}")
        exit(1)
        
    return tello

def setup_video_recorder(output_path, width=640, height=480, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
def robust_drone_control(func):
    """Decorator for robust drone command execution"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Command failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("All attempts failed, initiating emergency protocol")
                    return False
                time.sleep(0.5)
        return False
    return wrapper

# Apply to drone movements
@robust_drone_control
def safe_move_forward(drone, distance):
    drone.move_forward(distance)
    return True
def run_simulation_mode(detector, slam, rl_system):
    # Load a test video file or use webcam
    cap = cv2.VideoCapture(0)  # Use webcam as fallback
    
    # Try to load a test video if available
    test_videos = ['tunnel_test.mp4', 'underground_test.mp4', 'test_footage.mp4']
    for video in test_videos:
        if os.path.exists(video):
            cap = cv2.VideoCapture(video)
            print(f"Using test video: {video}")
            break
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Create windows for map views
    cv2.namedWindow("3D Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3D Map", 400, 400)
    cv2.namedWindow("2D Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("2D Map", 400, 400)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # If video ends, loop it by resetting the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            
            # Process with the RL system (which integrates detection and SLAM)
            processed_frame, action = rl_system.process_frame(frame)
            
            # Generate and display map visualizations (every 5 frames to reduce computation)
            if frame_count % 5 == 0:
                # 3D Map visualization
                map_3d = slam.visualize_3d_map()
                if map_3d is not None:
                    cv2.imshow("3D Map", map_3d)
                
                # 2D Map visualization
                map_2d = slam.visualize_2d_map((400, 400))
                if map_2d is not None:
                    cv2.imshow("2D Map", map_2d)
            
            # Display frame
            cv2.imshow("Underground Drone System (Simulation)", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to control frame rate
            time.sleep(0.05)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
class ThreadedDroneSystem:
    def __init__(self, detector, slam, rl_system):
        self.detector = detector
        self.slam = slam
        self.rl_system = rl_system
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        
    def detection_worker(self):
        """Run detection in separate thread"""
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.detector.detect(frame)
                self.result_queue.put(results)
                
    def start_threaded_processing(self):
        detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        detection_thread.start()

class EmergencyProtocol:
    def __init__(self, drone, slam):
        self.drone = drone
        self.slam = slam
        self.emergency_triggered = False
        
    def check_emergency_conditions(self, telemetry):
        """Check for emergency conditions"""
        conditions = {
            'low_battery': telemetry.get('battery', 100) < 15,
            'connection_lost': not self.drone.is_connected,
            'extreme_altitude': abs(telemetry.get('height', 0)) > 300,
            'no_safe_directions': len(self.slam.safe_directions) == 0
        }
        
        return any(conditions.values()), conditions
    
    def execute_emergency_landing(self):
        """Safe emergency landing procedure"""
        if not self.emergency_triggered:
            self.emergency_triggered = True
            print("EMERGENCY PROTOCOL ACTIVATED")
            
            try:
                # Try controlled landing first
                self.drone.land()
            except:
                try:
                    # Emergency motor stop as last resort
                    self.drone.emergency()
                except:
                    print("CRITICAL: Unable to execute emergency procedures")
def main():
    args = parse_args()
    
    # Create analysis directory if it doesn't exist
    if not os.path.exists(args.analysis_dir):
        os.makedirs(args.analysis_dir)
        print(f"Created analysis directory: {args.analysis_dir}")
    
    # Check if trajectory video generation is requested
    if args.gen_video:
        if os.path.exists(args.gen_video):
            print(f"Generating video from trajectory {args.gen_video}...")
            # Try to find corresponding map file
            map_file = args.gen_video.replace('trajectory', 'map')
            if not os.path.exists(map_file):
                print("Warning: Map file not found. Video will only show trajectory.")
                map_file = None
            
            video_output = os.path.join(args.analysis_dir, f"trajectory_video_{os.path.basename(args.gen_video)}.mp4")
            trajectory_video_generator(args.gen_video, map_file, output_file=video_output)
        else:
            print(f"Trajectory file not found: {args.gen_video}")
        return  # Exit after video generation
    
    # Check if trajectory comparison is requested
    if args.compare_trajectories:
        if len(args.compare_trajectories) < 2:
            print("Error: At least two trajectory files are needed for comparison")
            return
        
        # Verify all files exist
        all_exist = True
        for traj_file in args.compare_trajectories:
            if not os.path.exists(traj_file):
                print(f"Trajectory file not found: {traj_file}")
                all_exist = False
        
        if all_exist:
            print(f"Comparing {len(args.compare_trajectories)} trajectories...")
            
            # Use provided labels or generate default ones
            labels = args.trajectory_labels if args.trajectory_labels else None
            
            # Generate comparison output
            comparison_output = os.path.join(args.analysis_dir, "trajectory_comparison.png")
            compare_trajectories(args.compare_trajectories, labels, output_file=comparison_output)
        
        return  # Exit after comparison
    
    # Check if visualization is requested
    if args.vis_data:
        if os.path.exists(args.vis_data):
            print(f"Visualizing saved trajectory from {args.vis_data}")
            # Try to find corresponding map file
            map_file = args.vis_data.replace('trajectory', 'map')
            if not os.path.exists(map_file):
                map_file = None
            
            output_file = os.path.join(args.analysis_dir, f"visualization_{os.path.basename(args.vis_data)}.png")
            total_distance = visualize_trajectory(args.vis_data, map_file, output_file=output_file, show_analysis=True)
            
            # Generate a basic report
            report_file = os.path.join(args.analysis_dir, f"report_{os.path.basename(args.vis_data)}.txt")
            
            # Create basic metadata for standalone visualization
            metadata = {
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "trajectory_file": args.vis_data,
                "map_file": map_file if map_file else "Not available",
                "trajectory_length": len(np.load(args.vis_data)),
                "map_resolution": 10,  # Default value
                "map_size": [1000, 1000],  # Default value
                "home_position": [0, 0, 0]  # Default value
            }
            
            generate_mission_report(metadata, total_distance, report_file)
        else:
            print(f"Trajectory file not found: {args.vis_data}")
        return  # Exit after visualization

    if args.vis_metadata:
        if os.path.exists(args.vis_metadata):
            print(f"Visualizing mission data from {args.vis_metadata}")
            load_and_display_mission_data(args.vis_metadata, output_dir=args.analysis_dir)
        else:
            print(f"Metadata file not found: {args.vis_metadata}")
        return  # Exit after visualization
    
    # Initialize components
    print("Initializing Human Detection System...")
    detector = HumanDetector(args.model, device=args.device)
    
    print("Initializing SLAM System...")
    slam = SLAMSystem(min_obstacle_distance=50)  # 50cm minimum obstacle distance
    
    # Setup drone or simulation
    if args.sim:
        print("Running in simulation mode...")
        # Create a dummy drone object for simulation
        class DummyDrone:
            def __init__(self):
                self.is_flying = False
            def get_battery(self):
                return 100
            def get_height(self):
                return 100
            def get_attitude(self):
                return {"roll": 0, "pitch": 0, "yaw": 0}
            def takeoff(self):
                self.is_flying = True
                print("Dummy drone takeoff")
            def land(self):
                self.is_flying = False
                print("Dummy drone land")
            def move_forward(self, val):
                print(f"Dummy drone move forward {val}cm")
            def move_back(self, val):
                print(f"Dummy drone move back {val}cm")
            def move_left(self, val):
                print(f"Dummy drone move left {val}cm")
            def move_right(self, val):
                print(f"Dummy drone move right {val}cm")
            def move_up(self, val):
                print(f"Dummy drone move up {val}cm")
            def move_down(self, val):
                print(f"Dummy drone move down {val}cm")
        
        tello = DummyDrone()
    else:
        print("Connecting to Tello drone...")
        tello = setup_drone()
    
    # Initialize RL system (integrates all components)
    print("Initializing Reinforcement Learning System...")
    rl_system = DroneRLSystem(detector, slam, tello)
    
    # Load pre-trained RL model if specified
    if args.rl_model and os.path.exists(args.rl_model):
        print(f"Loading RL model from {args.rl_model}...")
        rl_system.agent.load_model(args.rl_model)
    
    # Setup video recorder if needed
    video_writer = None
    if args.record:
        print(f"Recording video to {args.output}...")
        video_writer = setup_video_recorder(args.output)
    
    try:
        print("Starting mission...")
        if args.sim:
            # Run in simulation mode (no actual drone)
            run_simulation_mode(detector, slam, rl_system)
        else:
            # Take off
            if not tello.is_flying:
                tello.takeoff()
                time.sleep(1)
            
            # Run the mission with the drone
            if args.training:
                # Run training mode (multiple episodes)
                print("Running in training mode...")
                rl_system.run(num_episodes=20)  # Run for 20 episodes
            else:
                # Run inference mode (single mission)
                print("Running in inference mode...")
                frame_count = 0

                while True:
                    # Get frame from the Tello drone
                    frame = tello.get_frame_read().frame
                    if frame is None or frame.size == 0:
                        print("Invalid frame received.")
                        time.sleep(0.1)
                        continue
                    
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    frame_count += 1
                    
                    # Process with the RL system
                    processed_frame, action = rl_system.process_frame(frame)
                    
                    # Execute action (but don't train)
                    if action is not None:
                        rl_system.execute_action(action)
                    
                    # Record video if enabled
                    if video_writer:
                        video_writer.write(processed_frame)
                    
                    # Display frame and check for key presses
                    cv2.imshow("Underground Drone System", processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("Mission interrupted by user")
                        break
                    elif key == ord('h'):
                        print("Return to home command received")
                        return_to_home(tello, slam, processed_frame.shape[1], processed_frame.shape[0])
                    elif key == ord('s'):
                        print("Saving current mission data...")
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        slam.save_all_data(f"manual_save_{timestamp}")
                    elif key == ord('v'):
                        # Quick visualization of current trajectory
                        print("Generating quick visualization of current trajectory...")
                        temp_traj_file = os.path.join(args.analysis_dir, "temp_trajectory.npy")
                        np.save(temp_traj_file, np.array(slam.pose_history))
                        visualize_trajectory(temp_traj_file, show_plot=True, show_analysis=True)
                    
                    # Small delay to control frame rate
                    time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("Mission interrupted by user")
    
    finally:
        print("Mission complete, landing drone...")
        # Land the drone with better error handling
        if not args.sim and tello.is_flying:
            try:
                # Try normal landing
                tello.land()
            except Exception as e:
                print(f"Error during landing: {e}")
                try:
                    # Try emergency landing as a fallback
                    print("Attempting emergency stop...")
                    tello.emergency()
                except:
                    print("WARNING: Could not land drone properly. Please manually power off the drone.")
        
        # Save maps as images with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("Saving maps...")
        slam.save_maps(os.path.join(args.analysis_dir, f"mission_maps_{timestamp}"))
        
        # Save mission data if requested
        if args.save_data:
            print("Saving raw mission data...")
            mission_data_path = os.path.join(args.analysis_dir, f"mission_data_{timestamp}")
            slam.save_all_data(mission_data_path)
            
            # Generate trajectory video automatically
            if len(slam.pose_history) > 10:
                traj_file = f"{mission_data_path}_trajectory_{timestamp}.npy"
                map_file = f"{mission_data_path}_map_{timestamp}.npy"
                
                if os.path.exists(traj_file) and os.path.exists(map_file):
                    video_path = os.path.join(args.analysis_dir, f"mission_video_{timestamp}.mp4")
                    print(f"Generating trajectory video: {video_path}")
                    trajectory_video_generator(traj_file, map_file, output_file=video_path)
        
        # Cleanup
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Generate mission summary report
        report_path = os.path.join(args.analysis_dir, f"mission_summary_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("=== UNDERGROUND DRONE MISSION SUMMARY ===\n\n")
            f.write(f"Mission Date/Time: {timestamp}\n")
            f.write(f"Mission Duration: {len(slam.pose_history) / 5:.2f} seconds\n\n")
            f.write(f"Humans detected: {'Yes' if rl_system.human_detected else 'No'}\n")
            f.write(f"Collisions occurred: {'Yes' if rl_system.collision_occurred else 'No'}\n")
            f.write(f"Mission complete: {'Yes' if rl_system.mission_complete else 'No'}\n\n")
            
            # Calculate total distance traveled
            total_distance = 0
            if len(slam.pose_history) > 1:
                for i in range(1, len(slam.pose_history)):
                    total_distance += np.linalg.norm(
                        np.array(slam.pose_history[i]) - np.array(slam.pose_history[i-1])
                    )
            f.write(f"Total distance traveled: {total_distance:.2f} cm\n")
            f.write(f"Maximum altitude: {max([p[2] for p in slam.pose_history]) if slam.pose_history else 0:.2f} cm\n")
            f.write(f"Average altitude: {sum([p[2] for p in slam.pose_history])/len(slam.pose_history) if slam.pose_history else 0:.2f} cm\n\n")
            
            f.write("System configuration:\n")
            f.write(f"- Detection model: {args.model}\n")
            f.write(f"- Detection interval: {args.detection_interval} frames\n")
            f.write(f"- Minimum obstacle distance: {slam.min_obstacle_distance} cm\n")
            
            if args.save_data:
                f.write("\nSaved data files:\n")
                f.write(f"- Maps: {os.path.join(args.analysis_dir, f'mission_maps_{timestamp}')}\n")
                f.write(f"- Mission data: {mission_data_path}\n")
                if os.path.exists(video_path):
                    f.write(f"- Trajectory video: {video_path}\n")
        
        print(f"Mission summary saved to {report_path}")
        print("System shutdown complete.")


if __name__ == "__main__":
    main()