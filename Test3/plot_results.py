import numpy as np
import matplotlib.pyplot as plt
import struct
from matplotlib.animation import FuncAnimation

def load_simulation_data(filename):
    """Load binary data from CUDA simulation"""
    with open(filename, 'rb') as f:
        # Read metadata
        metadata = np.fromfile(f, dtype=np.int32, count=3)
        num_simulations, num_steps, state_dim = metadata
        
        # Read time array
        time = np.fromfile(f, dtype=np.float32, count=num_steps + 1)
        
        # Read input signal
        u = np.fromfile(f, dtype=np.float32, count=num_steps)
        
        # Read state history
        x_history = np.fromfile(f, dtype=np.float32)
        x_history = x_history.reshape((num_simulations, num_steps + 1, state_dim))
    
    return time, u, x_history

def plot_results():
    # Load data
    time, u, x_history = load_simulation_data('simulation_results.bin')
    num_simulations = x_history.shape[0]
    
    print(f"Loaded {num_simulations} simulations")
    print(f"Time steps: {len(time)}")
    print(f"State dimension: {x_history.shape[2]}")
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot input signal
    ax1.step(time[:-1], u, where='post', color='red', linewidth=2)
    ax1.set_ylabel('Input u(t)')
    ax1.set_title('Input Signal')
    ax1.grid(True)
    
    # Plot a few representative trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, min(10, num_simulations)))
    
    for i in range(min(10, num_simulations)):
        ax2.plot(time, x_history[i, :, 0], color=colors[i], alpha=0.7, 
                label=f'Sim {i+1}' if i < 5 else "")
        ax3.plot(time, x_history[i, :, 1], color=colors[i], alpha=0.7,
                label=f'Sim {i+1}' if i < 5 else "")
    
    ax2.set_ylabel('Position x(t)')
    ax2.set_title('Position Trajectories')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_ylabel('Velocity v(t)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Velocity Trajectories')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot statistics
    plot_statistics(time, x_history)

def plot_statistics(time, x_history):
    """Plot statistical properties of all simulations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calculate statistics
    mean_position = np.mean(x_history[:, :, 0], axis=0)
    std_position = np.std(x_history[:, :, 0], axis=0)
    
    mean_velocity = np.mean(x_history[:, :, 1], axis=0)
    std_velocity = np.std(x_history[:, :, 1], axis=0)
    
    # Plot mean ± std
    ax1.fill_between(time, mean_position - std_position, mean_position + std_position, 
                    alpha=0.3, color='blue', label='±1 STD')
    ax1.plot(time, mean_position, 'b-', linewidth=2, label='Mean Position')
    ax1.set_ylabel('Position')
    ax1.set_title('Statistical Analysis of 10,000 Simulations')
    ax1.grid(True)
    ax1.legend()
    
    ax2.fill_between(time, mean_velocity - std_velocity, mean_velocity + std_velocity,
                    alpha=0.3, color='red', label='±1 STD')
    ax2.plot(time, mean_velocity, 'r-', linewidth=2, label='Mean Velocity')
    ax2.set_ylabel('Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_results()