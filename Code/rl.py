import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import cv2
import time
from typing import List, Tuple, Dict, Any

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_freq = 10  # Frequency to update target network
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer()
        
        # Training parameters
        self.steps_done = 0
        self.total_reward = 0
        self.episode_count = 0
        
        # Action mapping
        self.action_map = self._create_action_map()
        
    def _create_action_map(self):
        # Define 6 basic actions: forward, backward, left, right, up, down
        return {
            0: {"command": "forward", "value": 20},    # Move forward 20cm
            1: {"command": "back", "value": 20},       # Move backward 20cm
            2: {"command": "left", "value": 20},       # Move left 20cm
            3: {"command": "right", "value": 20},      # Move right 20cm
            4: {"command": "up", "value": 20},         # Move up 20cm
            5: {"command": "down", "value": 20},       # Move down 20cm
        }
    
    def get_state_from_environment(self, human_boxes, slam_data, drone_telemetry):
        # Initialize state vector
        state = []
        
        # 1. Drone position and orientation (normalized)
        if slam_data and "position" in slam_data:
            # Normalize position to [-1, 1] range based on map size
            map_size = 1000  # Assuming 10m x 10m map (1000cm)
            norm_pos = np.array(slam_data["position"]) / (map_size / 2)
            state.extend(list(norm_pos))
            
            # Normalize orientation to [-1, 1]
            norm_orientation = np.array(slam_data["orientation"]) / 180.0
            state.extend(list(norm_orientation))
        else:
            # If no SLAM data, use zeros
            state.extend([0, 0, 0, 0, 0, 0])
        
        # 2. Human detection features
        if human_boxes and len(human_boxes) > 0:
            # Get closest human (assume the largest bounding box is closest)
            largest_box = max(human_boxes, key=lambda box: box[2] * box[3])
            x, y, w, h = largest_box
            
            # Normalize box coordinates to [-1, 1]
            frame_width, frame_height = 320, 240  # From your main function
            norm_x = (x + w/2) / frame_width * 2 - 1  # Center X
            norm_y = (y + h/2) / frame_height * 2 - 1  # Center Y
            norm_size = (w * h) / (frame_width * frame_height)  # Relative size
            
            # Distance estimate (inverse of size)
            distance_estimate = 1.0 - norm_size
            
            state.extend([norm_x, norm_y, norm_size, distance_estimate])
            state.append(1.0)  # Human detected flag
        else:
            # No humans detected
            state.extend([0, 0, 0, 1.0, 0.0])
        
        # 3. Obstacle proximity in each direction (from SLAM)
        if slam_data and "safe_directions" in slam_data:
            # Convert safe directions to binary features
            directions = ["forward", "backward", "left", "right", "up", "down"]
            for direction in directions:
                state.append(1.0 if direction in slam_data["safe_directions"] else 0.0)
        else:
            # If no SLAM data, assume all directions are safe
            state.extend([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 4. Battery level (normalized to [0, 1])
        if drone_telemetry and "battery" in drone_telemetry:
            state.append(drone_telemetry["battery"] / 100.0)
        else:
            state.append(1.0)  # Assume full battery if no data
        
        return np.array(state, dtype=np.float32)
    
    def select_action(self, state):
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random action
            return random.randrange(self.action_size)
        else:
            # Exploit: select best action
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def get_reward(self, state, action, next_state, human_detected, collision_occured, battery_level, mission_complete):
        # Initialize reward
        reward = 0.0
        
        # Large penalties for collisions
        if collision_occured:
            reward -= 100.0
            return reward  # End episode on collision
        
        # Reward for detecting humans (primary mission objective)
        if human_detected:
            # Higher reward if human is centered in frame
            human_x, human_y = next_state[6], next_state[7]  # From state vector
            center_distance = np.sqrt(human_x**2 + human_y**2)
            
            # Max reward when human is centered (center_distance = 0)
            human_reward = 10.0 * (1.0 - min(center_distance, 1.0))
            reward += human_reward
        
        # Penalty for low battery (encourages efficient paths)
        battery_penalty = -0.1 * (1.0 - battery_level/100.0)
        reward += battery_penalty
        
        # Reward for mission completion
        if mission_complete:
            reward += 100.0
        
        # Small step penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def train(self):
        # Need enough samples in replay buffer
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path="drone_rl_model.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="drone_rl_model.pth"):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

class DroneRLSystem:
    def __init__(self, detector, slam, tello_drone):
        self.detector = detector  # HumanDetector instance
        self.slam = slam  # SLAMSystem instance
        self.drone = tello_drone  # Tello drone instance
        
        # Define state and action space
        self.state_size = 18  # Size of our state vector
        self.action_size = 6  # Number of possible actions
        
        # Initialize RL agent
        self.agent = DQNAgent(self.state_size, self.action_size)
        
        # Training parameters
        self.max_episode_steps = 1000
        self.train_frequency = 5  # Train every 5 steps
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_step = 0
        self.episode_count = 0
        
        # Current state and metrics
        self.current_state = None
        self.human_detected = False
        self.collision_occurred = False
        self.mission_complete = False
        
    def get_telemetry(self):
        try:
            telemetry = {
                "battery": self.drone.get_battery(),
                "height": self.drone.get_height(),
            }
            
            # Add additional telemetry if available
            # Note: Not all Tello models/SDK versions support these
            try:
                telemetry["temp"] = self.drone.get_temperature()
            except:
                pass
                
            try:
                telemetry["barometer"] = self.drone.get_barometer()
            except:
                pass
                
            try:
                telemetry["flight_time"] = self.drone.get_flight_time()
            except:
                pass
                
            return telemetry
            
        except Exception as e:
            print(f"Error getting telemetry: {e}")
            # Return default values if we can't get telemetry
            return {"battery": 100, "height": 100}
    
    # In the rl.py file, update the execute_action method
    def execute_action(self, action_idx):
        # Get the action details
        action = self.agent.action_map.get(action_idx)
        if not action:
            print(f"Unknown action index: {action_idx}")
            return False
        
        command = action["command"]
        value = action["value"]
        
        try:
            # Execute the command
            if command == "forward":
                self.drone.move_forward(value)
            elif command == "back":
                self.drone.move_back(value)
            elif command == "left":
                self.drone.move_left(value)
            elif command == "right":
                self.drone.move_right(value)
            elif command == "up":
                self.drone.move_up(value)
            elif command == "down":
                self.drone.move_down(value)
            return True
        except Exception as e:
            print(f"Error executing action: {e}")
            # Check if the error is related to motor issues
            if "Motor stop" in str(e):
                print("WARNING: Motor stop detected. The drone may need to be reset.")
                # You might want to initiate an emergency landing here
                try:
                    self.drone.emergency()  # Emergency stop - will drop the drone
                    print("Emergency stop initiated")
                except:
                    pass
            return False
    
    def check_for_collision(self, slam_data):

        
        # check if any safe direction exists
        if "safe_directions" in slam_data and len(slam_data["safe_directions"]) == 0:
            return True
        
        return False
    
    def check_mission_complete(self, human_detected, exploration_coverage=0.0):
        # For example, mission is complete if a human is detected and 50% of area explored
        return human_detected and exploration_coverage > 0.5
    
    def process_frame(self, frame):
        # Run human detection
        human_boxes = self.detector.detect(frame)
        self.human_detected = len(human_boxes) > 0
        
        # Run SLAM
        telemetry = self.get_telemetry()
        slam_frame, safe_directions = self.slam.process_frame(frame)
        
        # Create SLAM data dictionary
        slam_data = {
            "position": self.slam.position.tolist(),
            "orientation": self.slam.orientation.tolist(),
            "safe_directions": safe_directions
        }
        
        # Check for collision
        self.collision_occurred = self.check_for_collision(slam_data)
        
        # Check if mission is complete (simplified)
        self.mission_complete = self.check_mission_complete(self.human_detected)
        
        # Get RL state
        next_state = self.agent.get_state_from_environment(human_boxes, slam_data, telemetry)
        
        # Select action
        action = None
        if self.current_state is not None:
            # Get reward for previous action
            reward = self.agent.get_reward(
                self.current_state, 
                self.last_action,
                next_state, 
                self.human_detected,
                self.collision_occurred,
                telemetry["battery"],
                self.mission_complete
            )
            
            # Add to current episode reward
            self.current_episode_reward += reward
            
            # Store experience in replay buffer
            done = self.collision_occurred or self.mission_complete or self.episode_step >= self.max_episode_steps
            self.agent.memory.add(self.current_state, self.last_action, reward, next_state, done)
            
            # Train the agent periodically
            if self.episode_step % self.train_frequency == 0:
                self.agent.train()
            
            # End episode if done
            if done:
                print(f"Episode {self.episode_count} finished after {self.episode_step} steps")
                print(f"Episode reward: {self.current_episode_reward}")
                self.episode_rewards.append(self.current_episode_reward)
                
                # Reset episode
                self.current_episode_reward = 0
                self.episode_step = 0
                self.episode_count += 1
                
                # Save model periodically
                if self.episode_count % 10 == 0:
                    self.agent.save_model(f"drone_rl_model_ep{self.episode_count}.pth")
            
            # Select action for next step
            action = self.agent.select_action(next_state)
            self.last_action = action
        else:
            # First frame, select random action
            action = random.randrange(self.action_size)
            self.last_action = action
        
        # Update current state
        self.current_state = next_state
        self.episode_step += 1
        
        # Draw RL info on frame
        processed_frame = self._draw_rl_info(slam_frame, action, self.current_episode_reward)
        
        return processed_frame, action
    
    def _draw_rl_info(self, frame, action, episode_reward):
        # Clone frame for drawing
        vis_frame = frame.copy()
        
        # Draw action
        action_text = f"Action: {self.agent.action_map[action]['command']}"
        cv2.putText(vis_frame, action_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw episode info
        episode_text = f"Episode: {self.episode_count}, Step: {self.episode_step}"
        cv2.putText(vis_frame, episode_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw reward
        reward_text = f"Reward: {episode_reward:.1f}"
        cv2.putText(vis_frame, reward_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw human detection status
        human_text = "Human Detected" if self.human_detected else "No Human"
        color = (0, 255, 0) if self.human_detected else (0, 0, 255)
        cv2.putText(vis_frame, human_text, (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw exploration progress (simplified)
        cv2.putText(vis_frame, "Learning...", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return vis_frame
    
    def run(self, num_episodes=100):
        # Initialize drone if not already done
        if not self.drone.is_flying:
            self.drone.takeoff()
        
        try:
            while self.episode_count < num_episodes:
                # Get frame from drone
                frame = self.drone.get_frame_read().frame
                if frame is None or frame.size == 0:
                    print("Invalid frame received.")
                    time.sleep(0.1)
                    continue
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (320, 240))
                
                # Process frame
                processed_frame, action = self.process_frame(frame)
                
                # Execute action
                if action is not None:
                    self.execute_action(action)
                
                # Display frame
                cv2.imshow("Drone RL System", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small delay to control frame rate
                time.sleep(0.05)
                
        finally:
            # Land drone
            self.drone.land()
            cv2.destroyAllWindows()
            
            # Save final model
            self.agent.save_model("drone_rl_model_final.pth")
            
            # Plot training results
            self._plot_training_results()
    
    def _plot_training_results(self):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig('drone_rl_training_results.png')
        plt.close()