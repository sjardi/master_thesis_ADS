import pandas as pd
import re

# Load the log file
log_file_path = 'results.log'  # Ensure this is the correct path
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# Extract combination, start time, end time, and compute time taken
data = []
combination = None
start_time = None
end_time = None

# Define a mapping for cleaner names
name_mapping = {
    'sbert_zscore_minmax': 'z-score - minmax',
    'sbert_pareto_minmax': 'pareto - minmax',
    'sbert_l2_normalize_minmax': 'l2 normalize - minmax',
    'sbert_zscore_absmin': 'z-score - absmin',
    'sbert_pareto_absmin': 'pareto - absmin',
    'sbert_l2_normalize_absmin': 'l2 normalize - absmin',
    'sbert_zscore_sigmoid': 'z-score - sigmoid',
    'sbert_pareto_sigmoid': 'pareto - sigmoid',
    'sbert_l2_normalize_sigmoid': 'l2 normalize - sigmoid',
    'sbert_zscore_sqrt': 'z-score - sqrt',
    'sbert_pareto_sqrt': 'pareto - sqrt',
    'sbert_l2_normalize_sqrt': 'l2 normalize - sqrt',
    'sbert_zscore_cdf': 'z-score - cdf',
    'sbert_pareto_cdf': 'pareto - cdf',
    'sbert_l2_normalize_cdf': 'l2 normalize - cdf',
    'sbert_sigmoid': 'sigmoid',
    'sbert_cdf': 'cdf',
    'sbert_minmax': 'minmax',
    'sbert_absmin': 'absmin',
    'sbert': 'sbert'
    # Add other necessary mappings here
}

for line in log_lines:
    if 'Running classifier' in line:
        try:
            print(f"Processing line: {line.strip()}")
            combination_raw = line.split('Running classifier: ')[1].strip()
            # Extract and map the feature extractor part
            parts = combination_raw.split(' with feature extractor: ')
            classifier = parts[0].strip()
            feature_extractor = parts[1].strip() if len(parts) > 1 else 'unknown'
            combination = name_mapping.get(feature_extractor, feature_extractor)
            print(f"Combination: {classifier} - {combination}")
        except IndexError:
            print(f"Failed to process line: {line.strip()}")
            combination = None
            continue
    elif 'Start Time' in line:
        start_time = line.split('Start Time: ')[1].strip()
        print(f"Start Time: {start_time}")
    elif 'End Time' in line:
        end_time = line.split('End Time: ')[1].strip()
        print(f"End Time: {end_time}")
        if combination and start_time and end_time:
            # Calculate time taken
            time_taken = pd.to_datetime(end_time, format='%H:%M:%S,%f') - pd.to_datetime(start_time, format='%H:%M:%S,%f')
            data.append([combination, time_taken.total_seconds()])
            combination = start_time = end_time = None

# Create DataFrame for the extracted timing data
timing_df = pd.DataFrame(data, columns=['Combination', 'Time Taken (s)'])

# Display the DataFrame
print(timing_df)

# If you need to save this to a CSV file, uncomment the next line
timing_df.to_csv('sbert_timing_data.csv', index=False)