import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Conversion factor: Pixels to millimeters
pixels_to_mm = 19.5 / 512

# Function to calculate the median pupil diameter for each trial
def calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time, sample_rate):
    medians = []
    valid_indices = []
    for idx, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        if stop_sample <= len(pupil_diameter_pixels):
            trial_data = pupil_diameter_pixels[start_sample:stop_sample]
            median_diameter = np.median(trial_data) * pixels_to_mm
            medians.append(median_diameter)
            valid_indices.append(idx)
    return np.array(medians), valid_indices

# Function to get spike counts
def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    return np.array(baseline_counts), np.array(evoked_counts)

# Initialize combined data for plotting
all_low_trials = []
all_high_trials = []
all_low_firing_rates = []
all_high_firing_rates = []

# Process multiple datasets
for dataset_num in range(1, 3):  # Loop through the  datasets (1 to 6)
    print(f"Processing dataset {dataset_num}...")

    # Load data for each dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])
    pupil_diameter_voltage_path = utils.getFilePath(windowTitle=f"Pupil Diameter Data Dataset {dataset_num}", filetypes=[('Numpy file', '*.npy')])
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])

    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    trialDF = pd.read_csv(trial_DF_path)
    pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

    # Normalize pupil voltage and convert to pixels
    pupil_diameter_voltage = pupil_diameter_voltage / 1000
    beta = 511 / 10  # Voltage range (-5V to 5V)
    alpha = 1 - beta * (-5)
    pupil_diameter_pixels = beta * pupil_diameter_voltage + alpha

    # Extract metadata
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)

    # Calculate median pupil diameter for each trial
    trial_medians, valid_indices = calculate_median_pupil_diameter(
        pupil_diameter_pixels, stimulusDF, baseline_time=0.15, sample_rate=sRate
    )
    stimulusDF = stimulusDF.iloc[valid_indices].reset_index(drop=True)

    # Calculate pupil areas
    pupil_areas = (np.pi * (trial_medians ** 2)) / 4
    median_pupil_area = np.median(pupil_areas)

    # Split trials into low and high pupil area groups
    low_trials = pupil_areas[pupil_areas < median_pupil_area]
    high_trials = pupil_areas[pupil_areas >= median_pupil_area]

    # Calculate evoked firing rates for low and high trials
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)
    firing_rates_low = []
    firing_rates_high = []

    for cluster in spike_times_clusters.keys():
        spike_times = spike_times_clusters[cluster]

        low_indices = stimulusDF.index[pupil_areas < median_pupil_area]
        high_indices = stimulusDF.index[pupil_areas >= median_pupil_area]

        baseline_counts_low, evoked_counts_low = get_spike_counts(
            spike_times, stimulusDF.iloc[low_indices]['stimstart'], pre_time=0.15, post_time=0.15
        )
        baseline_counts_high, evoked_counts_high = get_spike_counts(
            spike_times, stimulusDF.iloc[high_indices]['stimstart'], pre_time=0.15, post_time=0.15
        )

        firing_rates_low.extend(evoked_counts_low / 0.15)
        firing_rates_high.extend(evoked_counts_high / 0.15)

    # Aggregate results
    all_low_trials.extend(low_trials)
    all_high_trials.extend(high_trials)
    all_low_firing_rates.extend([np.mean(firing_rates_low)] * len(low_trials))
    all_high_firing_rates.extend([np.mean(firing_rates_high)] * len(high_trials))

# Plot combined Pupil Area vs. Firing Rate for all datasets
plt.figure(figsize=(12, 8))
plt.scatter(all_low_trials, all_low_firing_rates, color='blue', label='Low Trials', alpha=0.7)
plt.scatter(all_high_trials, all_high_firing_rates, color='red', label='High Trials', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Plot formatting
plt.xlabel('Pupil Area (mm²)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Firing Rate (Hz)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



