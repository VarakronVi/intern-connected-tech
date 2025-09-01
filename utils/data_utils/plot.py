import sys
import os
import matplotlib.pyplot as plt
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from multiprocessing import Process, Queue
from src.calculation_layer.kalman_filter_manager import KalmanFilterManager
from utils.io_utils.config_handler import conf

def get_mock_data():
    pass

def producer(output_queues):
    """Simulate or receive measurements and put them into the queues."""
    while True:
        for i, queue in enumerate(output_queues):
            json_data = {
                "tag_id": f"tag_{i}",
                "timestamp": "2024-09-28T15:37:58.666932+07:00",
                "data": {
                    "uwb_dist": {
                        "dist_timestamp": "2024-09-28T15:37:58.666932+07:00",
                        "dist": [
                            {"anchor_id": "8121", "distance": 5.64},
                            {"anchor_id": "1BA7", "distance": 6.49},
                            {"anchor_id": "CE1A", "distance": 2.23},
                            {"anchor_id": "D387", "distance": 0.8}
                        ]
                    },
                    "uwb_acc": [
                        {
                            "acc_timestamp": "2024-09-28T15:37:58.666932+07:00",
                            "acc": {"x": 16032.0, "y": 928.0, "z": 3456.0}
                        },
                        {
                            "acc_timestamp": "2024-09-28T15:37:58.666940+07:00",
                            "acc": {"x": 16032.0, "y": 928.0, "z": 3456.0}
                        }
                    ]
                }
            }
            queue.put(json_data)
        time.sleep(1.0)


def processor(input_queues, state_queue):
    """Process measurements from the queues and update the Kalman filter."""
    kalman_filters = {f"tag_{i}": KalmanFilterManager(state_dim=4, measurement_dim=2) for i in range(len(input_queues))}

    while True:
        for i, queue in enumerate(input_queues):
            if not queue.empty():
                json_data = queue.get()
                tag_id = json_data['tag_id']
                kf_manager = kalman_filters[tag_id]
                kf_manager.run(json_data)
                current_state = kf_manager.get_current_state()
                state_queue.put((tag_id, current_state))


def calculate_total_distance(positions):
    """Helper function to calculate total Euclidean distance for a list of positions."""
    total_distance = 0.0
    if len(positions) > 1:
        for i in range(1, len(positions)):
            x1, y1 = positions[i - 1]
            x2, y2 = positions[i]
            total_distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return total_distance


def plot_real_time(state_queue):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    position_history = {}
    uwb_position_history = {}
    anchors = list(conf.UWB.ANCHOR_POSITIONS.values())  # Example anchor positions

    while True:
        tag_id, current_state, current_uwb_position = state_queue.get()
        print(f"Current state estimate for {tag_id}:")
        print(current_state)
        x = current_state[0][0]
        y = current_state[2][0]
        print(f"Current state for {tag_id}: ({x}, {y})")
        print(f"Current UWB position for {tag_id}: {current_uwb_position}")
        
        # Update the position history
        if tag_id not in position_history:
            position_history[tag_id] = []
        position_history[tag_id].append((x, y))
        
        # Update the UWB position history
        if tag_id not in uwb_position_history:
            uwb_position_history[tag_id] = []
        uwb_position_history[tag_id].append(current_uwb_position)
        
        # Clear the plot and redraw
        ax.clear()
        
        # Plot the position history as lines
        for tag, positions in position_history.items():
            positions_array = np.array(positions)
            ax.plot(positions_array[:, 0], positions_array[:, 1], label=f'{tag} Path')
            ax.scatter(positions_array[-1, 0], positions_array[-1, 1], label=f'{tag} Kalman Filter State', c='blue')

            # Calculate total distance for Kalman filtered position
            total_distance_position = calculate_total_distance(positions)
        
        # Plot the UWB positions
        for tag, uwb_positions in uwb_position_history.items():
            uwb_positions_array = np.array(uwb_positions)
            ax.scatter(uwb_positions_array[:, 0], uwb_positions_array[:, 1], label=f'{tag} UWB Position', c='green', marker='o')

            # Calculate total distance for UWB position
            total_distance_uwb = calculate_total_distance(uwb_positions)
        
        # Plot anchors
        anchors_x, anchors_y = zip(*anchors)
        ax.scatter(anchors_x, anchors_y, c='red', marker='x', label='Anchors')
        
        # Add labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Real-time Estimated Tag Positions')
        ax.grid(True)
        ax.legend()

        # Display total distances in the plot
        ax.text(0.02, 0.98, f'Total Kalman Distance: {total_distance_position:.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='blue')
        ax.text(0.02, 0.93, f'Total UWB Distance: {total_distance_uwb:.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='green')

        
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

if __name__ == "__main__":
    print(conf.UWB.ANCHOR_POSITIONS.values)
    measurement_queues = [Queue() for _ in range(2)]
    state_queue = Queue()

    producer_process = Process(target=producer, args=(measurement_queues,))
    processor_process = Process(target=processor, args=(measurement_queues, state_queue))

    producer_process.start()
    processor_process.start()

    try:
        plot_real_time(state_queue)
    except KeyboardInterrupt:
        print("Stopping processes...")
        producer_process.terminate()
        processor_process.terminate()
        producer_process.join()
        processor_process.join()

    print("All processes completed.")