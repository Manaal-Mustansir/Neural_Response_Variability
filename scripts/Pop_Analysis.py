import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Constants for penetration date and monkey name
penetration_date = '2024-05-29'
monkey_name = 'Sansa'

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)


# Loop through datasets
for dataset_num in range(1, 2):
    print(f"Processing dataset {dataset_num}...")

    # Load data for each dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])

    # Load spike times and cluster data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)
    # Load the trial mat file and extract stimuli
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']
   
    def linear_stimuli(expt_info):
        stims = np.array([])
        for i in range(expt_info.trial_records.shape[0]):
            stims = np.hstack((stims, expt_info.trial_records[i].trImage))
        return stims

    stimuli = linear_stimuli(expt_info)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]  # Removing NaN values from stimuli

    # Only keep "stimulus 1"
    stimulusDF = stimulusDF[stimulusDF['stimuli'] == 1]

    trialDF = pd.read_csv(trial_DF_path)
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter_voltage = np.load(pupilFullPath)

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

    # Define pupil areas for the dataset
    low_pupil_area = area_low
    high_pupil_area = area_high

    # Exclude trials near the median
    inds_low = low_trials
    inds_high = high_trials

    print(f"Number of low trials: {len(inds_low)}")
    print(f"Number of high trials: {len(inds_high)}")


    # Process clusters and spike times for analysis
    pre_time = 0.15
    post_time = 0.15
    initial_time = 0.035

    results = []

    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]

        # Only process "stimulus 1"
        stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == 1]
        # Load spike times and cluster data
        spike_times_sec = np.load(spike_times_secpath)
        clusters = np.load(clusters_path)
        # Get spike times clustered
        spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)
        # Process data for low and high groups excluding close-to-median trials
        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[inds_high]['stimstart'], pre_time, post_time)

        baseline_low_firing_rate = np.mean(baseline_counts_low) / pre_time
        baseline_high_firing_rate = np.mean(baseline_counts_high) / pre_time
        evoked_low_firing_rate = np.mean(evoked_counts_low) / post_time
        evoked_high_firing_rate = np.mean(evoked_counts_high) / post_time

        # Append results
        results.append({
            'Dataset': dataset_num,
            'Cluster': c,
            'Stimulus': 1,
            'Penetration': penetration_date,
            'Monkey Name': monkey_name,
            'Baseline Low Firing Rate': baseline_low_firing_rate,
            'Baseline High Firing Rate': baseline_high_firing_rate,
            'Evoked Low Firing Rate': evoked_low_firing_rate,
            'Evoked High Firing Rate': evoked_high_firing_rate,
            'Pupil Area Low': low_pupil_area,
            'Pupil Area High': high_pupil_area
            })

    # Convert results to a DataFrame for this dataset and save
    results_df = pd.DataFrame(results)
    csv_filename = f'results_dataset_{dataset_num}.csv'
    csv_fullpath = os.path.join(results_dir, csv_filename)
    results_df.to_csv(csv_fullpath, index=False)

    all_results.append(results_df)

# Combine results across all datasets


combined_results_df = pd.concat(all_results, ignore_index=True)
      
# Calculate means
mean_low_firing = np.mean(combined_results_df['Evoked Low Firing Rate'])
mean_high_firing = np.mean(combined_results_df['Evoked High Firing Rate'])

# Print the means
print(f"Mean Evoked Low Firing Rate: {mean_low_firing:.2f} Hz")
print(f"Mean Evoked High Firing Rate: {mean_high_firing:.2f} Hz")

# Save the means to a DataFrame
mean_firing_df = pd.DataFrame({
    "Metric": ["Mean Evoked Low Firing Rate", "Mean Evoked High Firing Rate"],
    "Value (Hz)": [mean_low_firing, mean_high_firing]
})

# Save the means DataFrame to a CSV
mean_csv_path = os.path.join(results_dir, "mean_firing_rates.csv")
mean_firing_df.to_csv(mean_csv_path, index=False)
print(f"Means saved to {mean_csv_path}")      
      
mean_low_area = np.mean(combined_results_df['Pupil Area Low'])
mean_high_area = np.mean(combined_results_df['Pupil Area High'])

mean_abs_firing_difference = abs(mean_high_firing - mean_low_firing)
print(f"Mean Absolute Firing: {mean_abs_firing_difference:.2f} Hz")

absolute_area_diff = abs(mean_high_area-mean_low_area)
print(f"Mean Absolute diff Area: {absolute_area_diff:.2f} mm^2")


plt.figure(figsize=(10, 6))
x_labels = [f'Low Pupil Area: {low_pupil_area:.2f} mm²', f'High Pupil Area: {high_pupil_area:.2f} mm²']
plt.bar(x_labels, [mean_low_firing, mean_high_firing], color=['gray', '#FF0000'], capsize=5, edgecolor='black', linewidth=3)

# Annotate absolute difference
plt.text(0.5, max(mean_low_firing, mean_high_firing) + 1, f'Abs Difference: {mean_abs_firing_difference:.2f} Hz',
         ha='center', fontsize=12, fontweight='bold')

plt.xlabel('Pupil Area (mm²)', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.title('Firing Rate by Pupil Area', fontsize=18, fontweight='bold')
plt.savefig(os.path.join(results_dir, 'firing_rate_by_pupil_area_barplot.svg'))
plt.show()

# Scatter Plot: 
#  absolute difference  pupil area 
absolute_area_diff = abs(combined_results_df['Pupil Area High'] - combined_results_df['Pupil Area Low'])


x_values = absolute_area_diff

# absolute differences  firing rates
absolute_differences = abs(combined_results_df['Evoked High Firing Rate'] - combined_results_df['Evoked Low Firing Rate'])


plt.figure(figsize=(8, 8))
plt.scatter(x_values, absolute_differences, c=combined_results_df['Dataset'], cmap='viridis', s=15)
plt.xlabel(' Pupil Area (mm²)', fontsize=16, fontweight='bold')
plt.ylabel(' Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.ylim(0, 25)  
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(results_dir, 'firing_rate_abs_diff_vs_pupil_area_diff_scatter.svg'))
plt.show()

import matplotlib.pyplot as plt

# Data
areas = [1.12, 1.03, 0.97, 0.83, 0.83, 0.79]  # Mean Absolute diff Area
firings = [6.59, 5.77, 5.29, 4.26, 3.69, 3.33]  # Mean Absolute Firing
labels = ['Penetration-1', 'Penetration-2', 'Penetration-3', 'Penetration-4', 'Penetration-5', 'Penetration-6']

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(areas, firings, color='blue', edgecolor='black', s=100)

# Add labels for each data point
for i, label in enumerate(labels):
    plt.text(areas[i], firings[i], label, fontsize=10, ha='right', va='bottom')

# Label axes with bold text
plt.xlabel(r'$\Delta$ Pupil Area (mm$^2$)', fontsize=20, fontweight='bold')
plt.ylabel(r'$\Delta$ Firing Rate (Hz)', fontsize=20, fontweight='bold')

# Customize ticks
plt.tick_params(axis='both', which='both', direction='in', length=10, width=2.5, labelsize=16)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')

# Customize axes borders (spines)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2.5)


# Grid for better readability
#plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig("scatter_plot.svg", format="svg")
plt.show()


