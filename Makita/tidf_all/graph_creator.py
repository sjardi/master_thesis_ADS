import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Path to the log file
log_file = 'system_metrics.log'

# Check if the log file exists
if not os.path.exists(log_file):
    print(f"Error: {log_file} not found.")
    exit(1)

# Lists to store the parsed data
mem_utils = []

# Parse the log file
with open(log_file, 'r') as file:
    for line in file:
        if "Raw output:" in line:
            value = line.split("Raw output:")[1].strip()
            if value.endswith('%'):
                mem_utils.append(int(value.replace('%', '').strip()))

# Ensure we have values to plot
if not mem_utils:
    print("Error: No valid memory utilization data found in the log file.")
    exit(1)

# Find the maximum value in the mem_utils list
max_value = max(mem_utils)

# Scale the values so that the maximum value becomes 100
scaled_utils = [int(value * 100 / max_value) for value in mem_utils]

# Function to adjust y values
def adjust_y(value):
    if value <= 20:
        return value
    elif value <= 60:
        return 20 + (value - 20) * 0.1  # Compress the 20-60% range
    else:
        return 24 + (value - 60)  # Shift the values above 60

# Adjust the scaled values
adjusted_utils = [adjust_y(value) for value in scaled_utils]

# Create a DataFrame
data = {
    'Time Step': list(range(len(adjusted_utils))),
    'Adjusted GPU Utilization (%)': adjusted_utils
}
df = pd.DataFrame(data)

# Plot GPU Utilization with adjusted y-values
plt.figure(figsize=(12, 6), dpi=120)
plt.plot(df['Time Step'], df['Adjusted GPU Utilization (%)'], label='GPU Utilization (%)', color='b')
plt.xlabel('Time Step')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.ylim(0, 64)  # Adjust y-axis limit to match the adjusted range
plt.grid(True, which='major', linestyle='--', linewidth=0.5)

# Custom y-ticks
yticks = np.array([0, 10, 20, 60, 70, 80, 90, 100])
yticks_adjusted = [adjust_y(tick) for tick in yticks]
plt.yticks(yticks_adjusted, yticks)

# Annotate the maximum value
max_index = df['Adjusted GPU Utilization (%)'].idxmax()
max_value = df['Adjusted GPU Utilization (%)'][max_index]
original_max_value = scaled_utils[max_index]
plt.annotate(f'Max: {original_max_value}%', 
             xy=(max_index, max_value), 
             xytext=(max_index, max_value + 2),
             horizontalalignment='center',
             fontsize=9,
             bbox=dict(facecolor='white', alpha=0.5))

# Save the figure
plt.savefig('gpu_utilization_squashed.png')

# Show the plot
plt.show()
