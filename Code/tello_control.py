from djitellopy import Tello

# Initialize the Tello drone
tello = Tello()

try:
    # Connect to the drone
    tello.connect()

    # Print battery percentage to confirm connection
    battery = tello.get_battery()
    print(f"Battery Life Percentage: {battery}%")

    # Test additional status commands if needed
    wifi_strength = tello.query_wifi_signal_noise_ratio()
    print(f"Wi-Fi Signal Strength: {wifi_strength}")

    # Optionally test video stream (ensure it starts correctly)
    tello.streamon()
    print("Video stream is active.")
    tello.streamoff()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Safely disconnect
    print("Test completed.")
 