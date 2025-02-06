import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data paths using utility functions
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])
pupil_diameter_voltage_path = utils.getFilePath(windowTitle="Pupil Diameter Data", filetypes=[('Numpy file', '*.npy')])

# Load actual data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

# Load binary nidq file and metadata
binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

# Voltage transformation parameters
min_voltage = -5
max_voltage = 5

# Normalize voltage
pupil_diameter_voltage = pupil_diameter_voltage / 1000

# Calculate beta and alpha for linear mapping
beta = 511 / (max_voltage - min_voltage)
alpha = 1 - beta * min_voltage

print(f"Linear Mapping Equation: pixels = {beta:.4f} * voltage + {alpha:.4f}")

# Apply linear transformation
pupil_diameter_pixels_mapped = beta * pupil_diameter_voltage + alpha

# Conversion factor: pixels to millimeters
pixels_to_mm = 19.5 / 512

# Function to calculate the median pupil diameter for each trial
def calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter_pixels[start_sample:stop_sample]
        medians.append(np.median(trial_data))
    return np.array(medians)

# Calculate median pupil diameters
trial_medians = calculate_median_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, baseline_time=0.15, sample_rate=sRate)

# Determine overall median and split trials into low and high
overall_median = np.median(trial_medians)
low_trials = np.where(trial_medians < overall_median)[0]
high_trials = np.where(trial_medians >= overall_median)[0]

# Function to calculate the average pupil diameter for specified trials
def calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, trial_indices, sample_rate):
    avg_pupil_diameters = []
    for trial in trial_indices:
        start_sample = int(stimulusDF.iloc[trial]['stimstart'] * sample_rate)
        stop_sample = int(stimulusDF.iloc[trial]['stimstop'] * sample_rate)
        avg_pupil_diameters.append(np.mean(pupil_diameter_pixels[start_sample:stop_sample]))
    return np.mean(avg_pupil_diameters)

# Calculate average pupil diameters for low and high trials
avg_low_pixels = calculate_average_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, low_trials, sRate)
avg_high_pixels = calculate_average_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, high_trials, sRate)

# Convert pixel values to millimeters
avg_low_mm = avg_low_pixels * pixels_to_mm
avg_high_mm = avg_high_pixels * pixels_to_mm

# Calculate pupil areas in square millimeters
area_low = (np.pi * (avg_low_mm ** 2)) / 4
area_high = (np.pi * (avg_high_mm ** 2)) / 4

print(f"Average Pupil Diameter for Low Trials: {avg_low_mm:.2f} mm")
print(f"Average Pupil Diameter for High Trials: {avg_high_mm:.2f} mm")
print(f"Estimated Pupil Area for Low Trials: {area_low:.2f} mm^2")
print(f"Estimated Pupil Area for High Trials: {area_high:.2f} mm^2")

# Plot voltage vs pupil diameter pixels
plt.figure(figsize=(10, 6))
plt.plot(pupil_diameter_voltage, pupil_diameter_pixels_mapped, label='Pupil Diameter (Pixels)', color='blue')
plt.xlabel('Voltage (V)', fontsize=14, fontweight='bold')
plt.ylabel('Pupil Diameter (Pixels)', fontsize=14, fontweight='bold')
plt.ylim(1, 249)
plt.title('Voltage vs Pupil Diameter (Pixels)', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

# Save the plot
output_plot_path = os.path.join(results_dir, 'voltage_vs_pupil_diameter_pixels.png')
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data paths using utility functions
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])
pupil_diameter_voltage_path = utils.getFilePath(windowTitle="Pupil Diameter Data", filetypes=[('Numpy file', '*.npy')])

# Load actual data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

# Load binary nidq file and metadata
binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

#
min_voltage = -5
max_voltage = 5

pupil_diameter_voltage = pupil_diameter_voltage/1000
# Calculate beta and alpha for the linear mapping equation
beta = 511 / (max_voltage - min_voltage)  # 
alpha = 1 - beta * min_voltage  # 

print(f"Linear Mapping Equation: pixels = {beta:.4f} * voltage + {alpha:.4f}")

# Apply the linear transformation: voltage -> pixels
pupil_diameter_pixels_mapped = beta * pupil_diameter_voltage + alpha

# Function to calculate the median pupil diameter for each trial
def calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter_pixels[start_sample:stop_sample]
        medians.append(np.median(trial_data))
    return np.array(medians)

# Calculate median pupil diameters
trial_medians = calculate_median_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, baseline_time=0.15, sample_rate=sRate)

# Determine overall median and split trials into low and high
overall_median = np.median(trial_medians)
low_trials = np.where(trial_medians < overall_median)[0]
high_trials = np.where(trial_medians >= overall_median)[0]

# Function to calculate the average pupil diameter for specified trials
def calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, trial_indices, sample_rate):
    avg_pupil_diameters = []
    for trial in trial_indices:
        start_sample = int(stimulusDF.iloc[trial]['stimstart'] * sample_rate)
        stop_sample = int(stimulusDF.iloc[trial]['stimstop'] * sample_rate)
        avg_pupil_diameters.append(np.mean(pupil_diameter_pixels[start_sample:stop_sample]))
    return np.mean(avg_pupil_diameters)

# Calculate average pupil diameter for low and high trials
avg_low = calculate_average_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, low_trials, sRate)
avg_high = calculate_average_pupil_diameter(pupil_diameter_pixels_mapped, stimulusDF, high_trials, sRate)

print(f"Average Pupil Diameter for Low Trials: {avg_low:.2f} ")
print(f"Average Pupil Diameter for High Trials: {avg_high:.2f}")

# Calculate voltage for low and high trials based on pixel values using the linear equation
voltage_low = (avg_low - alpha) / beta
voltage_high = (avg_high - alpha) / beta

print(f"Estimated Voltage for Low Trials: {voltage_low:.2f} V")
print(f"Estimated Voltage for High Trials: {voltage_high:.2f} V")

# Plot the voltage on the x-axis and pupil diameter in pixels on the y-axis
plt.figure(figsize=(10, 6))

# Plot voltage vs pupil diameter pixels
plt.plot(pupil_diameter_voltage, pupil_diameter_pixels_mapped, label='Pupil Diameter (Pixels)', color='blue')

# Set x and y labels
plt.xlabel('Voltage (V)', fontsize=14, fontweight='bold')
plt.ylabel('Pupil Diameter (Pixels)', fontsize=14, fontweight='bold')

# Set the y-axis limit to 1-512
plt.ylim(1, 249)


plt.title('Voltage vs Pupil Diameter (Pixels)', fontsize=16, fontweight='bold')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Save the figure
output_plot_path = os.path.join(results_dir, 'voltage_vs_pupil_diameter_pixels.png')
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data paths using utility functions
pupil_diameter_voltage_path = utils.getFilePath(windowTitle="Pupil Diameter Data", filetypes=[('Numpy file', '*.npy')])

# Load pupil diameter voltage data
pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

# Define the actual min and max voltage values (-5 to +5)
min_voltage = -5
max_voltage = 5

#
pupil_diameter_voltage = np.interp(pupil_diameter_voltage, (pupil_diameter_voltage.min(), pupil_diameter_voltage.max()), (min_voltage, max_voltage))

# Calculate beta and alpha for the linear mapping equation
beta = 511 / (max_voltage - min_voltage)  # 511 pixel range (512 - 1)
alpha = 1 - beta * min_voltage  # solving Eq. 2 for alpha

print(f"Linear Mapping Equation: pixels = {beta:.4f} * voltage + {alpha:.4f}")

# Apply the linear transformation: voltage -> pixels
pupil_diameter_pixels_mapped = beta * pupil_diameter_voltage + alpha

# Plot the voltage on the x-axis and pupil diameter in pixels on the y-axis
plt.figure(figsize=(10, 6))

# Plot voltage vs pupil diameter pixels
plt.plot(pupil_diameter_voltage, pupil_diameter_pixels_mapped, label='Pupil Diameter (Pixels)', color='blue')

# Set x and y labels
plt.xlabel('Voltage (V)', fontsize=14, fontweight='bold')
plt.ylabel('Pupil Diameter (Pixels)', fontsize=14, fontweight='bold')

# Set the y-axis limit to 1-512
plt.ylim(1, 512)

# Title and legend
plt.title('Voltage vs Pupil Diameter (Pixels)', fontsize=16, fontweight='bold')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# Save the figure
output_plot_path = os.path.join(results_dir, 'voltage_vs_pupil_diameter_pixels_corrected.png')
plt.savefig(output_plot_path)
print(f"Plot saved to {output_plot_path}")





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data paths using utility functions
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])
pupil_diameter_voltage_path = utils.getFilePath(windowTitle="Pupil Diameter Data", filetypes=[('Numpy file', '*.npy')])

# Load actual data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

min_voltage = pupil_diameter_voltage.min()
max_voltage = pupil_diameter_voltage.max()

print(f"Voltage Data Range: {min_voltage} to {max_voltage}")


normalized_voltage = (pupil_diameter_voltage - min_voltage) / (max_voltage - min_voltage)

pixel_horizontal = normalized_voltage * 255
pixel_vertical = normalized_voltage * 128


pupil_diameter_pixels = (pixel_horizontal + pixel_vertical) / 2

def calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter_pixels[start_sample:stop_sample]
        medians.append(np.median(trial_data))
    return np.array(medians)


trial_medians = calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time=0.15, sample_rate=sRate)

overall_median = np.median(trial_medians)
low_trials = np.where(trial_medians < overall_median)[0]
high_trials = np.where(trial_medians >= overall_median)[0]

def calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, trial_indices, sample_rate):
    avg_pupil_diameters = []
    for trial in trial_indices:
        start_sample = int(stimulusDF.iloc[trial]['stimstart'] * sample_rate)
        stop_sample = int(stimulusDF.iloc[trial]['stimstop'] * sample_rate)
        avg_pupil_diameters.append(np.mean(pupil_diameter_pixels[start_sample:stop_sample]))
    return np.mean(avg_pupil_diameters)


avg_low = calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, low_trials, sRate)
avg_high = calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, high_trials, sRate)

print(f"Average Pupil Diameter for Low Trials: {avg_low:.2f} pixels")
print(f"Average Pupil Diameter for High Trials: {avg_high:.2f} pixels")




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Conversion factor: Pixels to millimeters
pixels_to_mm = 19.5 / 512

# Function to calculate pupil areas for each trial
def calculate_trial_pupil_area(stimulusDF, pupil_diameter_mm, sample_rate):
    trial_areas = []
    for _, trial in stimulusDF.iterrows():
        start_sample = int(trial['stimstart'] * sample_rate)
        stop_sample = int(trial['stimstop'] * sample_rate)
        stop_sample = min(stop_sample, len(pupil_diameter_mm))  # Ensure stop_sample is within range
        trial_data = pupil_diameter_mm[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        area = (np.pi * (median_diameter ** 2)) / 4  # Calculate pupil area
        trial_areas.append(area)
    return np.array(trial_areas)

# Function to calculate mean firing rates
def calculate_mean_firing_rate(spike_times, trials, pre_time=0.15, post_time=0.15):
    mean_firing_rates = []
    for _, trial in trials.iterrows():
        start_time = trial['stimstart']
        evoked_spikes = spike_times[(spike_times >= start_time) & (spike_times < start_time + post_time)]
        mean_rate = len(evoked_spikes) / post_time
        mean_firing_rates.append(mean_rate)
    return np.mean(mean_firing_rates)

# Store results from all datasets
all_results = []

# Loop through datasets
for dataset_num in range(1, 3):  # Adjust range for your dataset count
    print(f"Processing Dataset {dataset_num}...")

    # Load data paths
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    pupil_diameter_voltage_path = utils.getFilePath(windowTitle=f"Pupil Diameter Data Dataset {dataset_num}", filetypes=[('Numpy file', '*.npy')])
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])

    # Load actual data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']
    stimulusDF = pd.read_csv(stimulus_DF_path)
    pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)

    # Normalize pupil diameter voltage
    pupil_diameter_voltage = pupil_diameter_voltage / 1000
    min_voltage, max_voltage = -5, 5
    beta = 511 / (max_voltage - min_voltage)
    alpha = 1 - beta * min_voltage
    pupil_diameter_pixels = beta * pupil_diameter_voltage + alpha
    pupil_diameter_mm = pupil_diameter_pixels * pixels_to_mm  # Convert pixels to mm

    # Extract stimuli for "stimulus 1" from the .mat file
    stimuli = []
    for trial in expt_info.trial_records:
        if isinstance(trial.trImage, (np.ndarray, list)):
            valid_images = trial.trImage[~np.isnan(trial.trImage)]  # Remove NaNs
            stimuli.extend(valid_images)
        elif not np.isnan(trial.trImage):
            stimuli.append(trial.trImage)
    stimuli = np.array(stimuli)  # Convert to NumPy array
    stimulusDF['stimuli'] = stimuli
    stimulusDF = stimulusDF[stimulusDF['stimuli'] == 1]

    # Compute pupil areas for trials
    trial_areas = calculate_trial_pupil_area(stimulusDF, pupil_diameter_mm, sRate)

    # Split trials into low and high based on median pupil area
    median_area = np.median(trial_areas)
    low_trials = stimulusDF[trial_areas < median_area]
    high_trials = stimulusDF[trial_areas >= median_area]

    # Analyze each cluster
    cluster_results = []
    for cluster_id in np.unique(clusters):
        cluster_spike_times = spike_times_sec[clusters == cluster_id]

        mean_firing_low = calculate_mean_firing_rate(cluster_spike_times, low_trials)
        mean_firing_high = calculate_mean_firing_rate(cluster_spike_times, high_trials)

        cluster_results.append({
            'Dataset': dataset_num,
            'Cluster': cluster_id,
            'Pupil Area Low': np.mean(trial_areas[trial_areas < median_area]),
            'Pupil Area High': np.mean(trial_areas[trial_areas >= median_area]),
            'Mean Firing Rate Low': mean_firing_low,
            'Mean Firing Rate High': mean_firing_high,
        })

    # Append results
    all_results.extend(cluster_results)

# Combine all results into a DataFrame
results_df = pd.DataFrame(all_results)

# Scatter plot: Pupil area vs. Mean firing rates for all clusters
plt.figure(figsize=(12, 8))
for dataset_num in results_df['Dataset'].unique():
    dataset_results = results_df[results_df['Dataset'] == dataset_num]
    plt.scatter(dataset_results['Pupil Area Low'], dataset_results['Mean Firing Rate Low'], color='blue', alpha=0.6)
    plt.scatter(dataset_results['Pupil Area High'], dataset_results['Mean Firing Rate High'], color='red', alpha=0.6)

# Format the plot
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Pupil Area (mm^2)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Firing Rate (Hz)', fontsize=14, fontweight='bold')
plt.title('Cluster-Wise Relationship Between Pupil Area and Mean Firing Rates', fontsize=16, fontweight='bold')
plt.grid(True)
plt.show()
