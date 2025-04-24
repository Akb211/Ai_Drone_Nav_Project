import numpy as np
import matplotlib.pyplot as plt
import os
import json
from mpl_toolkits.mplot3d import Axes3D
import cv2
from datetime import datetime
import matplotlib.cm as cm

def visualize_trajectory(trajectory_file, map_file=None, output_file='trajectory_visualization.png', 
                         show_analysis=True, show_plot=True):
    # Load trajectory
    trajectory = np.load(trajectory_file)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D subplot for trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='x', s=100, label='End')
    
    # Set labels and title
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # Calculate flight statistics for analysis display
    if show_analysis:
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(trajectory)):
            total_distance += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        # Calculate average and max altitude
        avg_altitude = np.mean(trajectory[:, 2])
        max_altitude = np.max(trajectory[:, 2])
        
        # Calculate flight duration (estimated from trajectory points)
        # Assuming average sampling rate of 5Hz (adjust as needed)
        est_duration = len(trajectory) / 5.0  # in seconds
        
        # Calculate average velocity
        avg_velocity = total_distance / est_duration if est_duration > 0 else 0
        
        # Display statistics in a text box
        ax3 = fig.add_subplot(223)
        ax3.axis('off')
        stats_text = (
            f"Flight Statistics:\n"
            f"------------------\n"
            f"Total distance: {total_distance:.2f} cm\n"
            f"Estimated duration: {est_duration:.2f} sec\n"
            f"Average altitude: {avg_altitude:.2f} cm\n"
            f"Maximum altitude: {max_altitude:.2f} cm\n"
            f"Average velocity: {avg_velocity:.2f} cm/s\n"
            f"Path efficiency: {np.linalg.norm(trajectory[-1] - trajectory[0])/total_distance*100:.1f}%\n"
        )
        ax3.text(0.1, 0.1, stats_text, fontsize=12, family='monospace')
    
    # If map file is provided, visualize the map
    if map_file and os.path.exists(map_file):
        # Load map
        occupancy_map = np.load(map_file)
        
        # 2D subplot for map
        ax2 = fig.add_subplot(222)
        ax2.imshow(occupancy_map.T, cmap='binary', origin='lower')
        
        # Convert trajectory to map coordinates
        map_size = 1000  # Assuming standard map size of 10m x 10m (1000cm)
        map_resolution = 10  # Assuming standard resolution of 10cm per cell
        
        traj_x = (trajectory[:, 0] + map_size/2) / map_resolution
        traj_y = (trajectory[:, 1] + map_size/2) / map_resolution
        
        # Plot trajectory on map with color indicating time progression
        points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a colormap for time progression
        norm = plt.Normalize(0, len(segments))
        lc = plt.LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(np.arange(len(segments)))
        lc.set_linewidth(2)
        line = ax2.add_collection(lc)
        fig.colorbar(line, ax=ax2, label='Time Progression')
        
        # Plot start and end points
        ax2.scatter(traj_x[0], traj_y[0], c='g', marker='o', s=100, label='Start')
        ax2.scatter(traj_x[-1], traj_y[-1], c='r', marker='x', s=100, label='End')
        
        # Set title and legend
        ax2.set_title('2D Occupancy Map with Trajectory')
        ax2.legend()
        
        # Add altitude profile subplot
        ax4 = fig.add_subplot(224)
        ax4.plot(range(len(trajectory)), trajectory[:, 2], 'g-')
        ax4.set_xlabel('Time (samples)')
        ax4.set_ylabel('Altitude (cm)')
        ax4.set_title('Altitude Profile')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return total_distance if show_analysis else None

def load_and_display_mission_data(metadata_file, output_dir='mission_analysis'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Print mission info
    print(f"Mission data from {metadata['timestamp']}")
    print(f"Map resolution: {metadata['map_resolution']} cm per cell")
    print(f"Map size: {metadata['map_size']} cm")
    print(f"Home position: {metadata['home_position']}")
    print(f"Trajectory length: {metadata['trajectory_length']} points")
    
    # Check if trajectory and map files exist
    traj_file = metadata['trajectory_file']
    map_file = metadata['map_file']
    pointcloud_file = metadata.get('pointcloud_file', None)
    
    if os.path.exists(traj_file):
        output_file = os.path.join(output_dir, f"analysis_{metadata['timestamp']}.png")
        
        if os.path.exists(map_file):
            # Visualize trajectory and map
            total_distance = visualize_trajectory(traj_file, map_file, output_file)
        else:
            # Visualize only trajectory
            total_distance = visualize_trajectory(traj_file, output_file=output_file)
        
        # Generate a mission report
        report_file = os.path.join(output_dir, f"mission_report_{metadata['timestamp']}.txt")
        generate_mission_report(metadata, total_distance, report_file)
        
    else:
        print("Trajectory file not found.")

def generate_mission_report(metadata, total_distance, output_file):
    with open(output_file, 'w') as f:
        f.write("=== UNDERGROUND DRONE MISSION REPORT ===\n\n")
        f.write(f"Mission Date/Time: {metadata['timestamp']}\n")
        f.write(f"Map Resolution: {metadata['map_resolution']} cm per cell\n")
        f.write(f"Map Size: {metadata['map_size'][0]}cm x {metadata['map_size'][1]}cm\n")
        f.write(f"Home Position: [{metadata['home_position'][0]}, {metadata['home_position'][1]}, {metadata['home_position'][2]}]\n")
        f.write(f"Trajectory Points: {metadata['trajectory_length']}\n")
        f.write(f"Total Distance: {total_distance:.2f} cm\n")
        
        # Estimated mission duration (assuming 5Hz data capture rate)
        est_duration = metadata['trajectory_length'] / 5.0
        f.write(f"Estimated Duration: {est_duration:.2f} seconds ({est_duration/60:.2f} minutes)\n")
        
        # Files generated
        f.write("\nAssociated Files:\n")
        f.write(f"- Trajectory: {metadata['trajectory_file']}\n")
        f.write(f"- Map: {metadata['map_file']}\n")
        if 'pointcloud_file' in metadata:
            f.write(f"- Point Cloud: {metadata['pointcloud_file']}\n")
        
        f.write("\n=== END OF REPORT ===\n")
    
    print(f"Mission report saved to {output_file}")

def trajectory_video_generator(trajectory_file, map_file, output_file='trajectory_video.mp4', fps=30):
    # Load trajectory and map
    trajectory = np.load(trajectory_file)
    occupancy_map = np.load(map_file)
    
    # Set up video writer
    frame_size = (800, 600)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    
    # Map parameters
    map_size = 1000
    map_resolution = 10
    
    # Create frames for each trajectory point
    for i in range(len(trajectory)):
        # Create figure for this frame
        fig = plt.figure(figsize=(8, 6), dpi=100)
        
        # Plot map
        plt.imshow(occupancy_map.T, cmap='binary', origin='lower')
        
        # Convert trajectory to map coordinates
        traj_x = (trajectory[:i+1, 0] + map_size/2) / map_resolution
        traj_y = (trajectory[:i+1, 1] + map_size/2) / map_resolution
        
        # Plot trajectory
        plt.plot(traj_x, traj_y, 'b-', linewidth=2)
        
        # Plot start and current position
        plt.scatter(traj_x[0], traj_y[0], c='g', marker='o', s=100)
        plt.scatter(traj_x[-1], traj_y[-1], c='r', marker='o', s=100)
        
        # Add title with progress indicator
        plt.title(f"Drone Trajectory - {i+1}/{len(trajectory)} frames")
        
        # Convert matplotlib figure to OpenCV image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, frame_size)
        
        # Write frame to video
        video.write(img)
        
        # Clean up
        plt.close(fig)
    
    # Release video writer
    video.release()
    print(f"Video saved to {output_file}")

def compare_trajectories(traj_files, labels=None, output_file='trajectory_comparison.png'):
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(traj_files))]
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D subplot (top-down view)
    ax2 = fig.add_subplot(122)
    
    # Plot each trajectory
    for i, traj_file in enumerate(traj_files):
        trajectory = np.load(traj_file)
        
        # Plot in 3D
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=2, label=labels[i])
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], marker='o', s=50)
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], marker='x', s=50)
        
        # Plot in 2D (top-down view)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, label=labels[i])
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], marker='o', s=50)
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], marker='x', s=50)
    
    # Set labels and titles
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    ax2.set_xlabel('X (cm)')
    ax2.set_ylabel('Y (cm)')
    ax2.set_title('2D Trajectory Comparison (Top-Down View)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)   
    print(f"Comparison saved to {output_file}")
    plt.show()