
#ORIGINAL
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

# Store results from all datasets
all_results = []

# Loop through six datasets
for dataset_num in range(1, 7):  # Loop through the desired dataset(s)
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
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    # Calculate median pupil diameter
    def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
        medians = []
        for index, row in stimulusDF.iterrows():
            start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
            stop_sample = int(row['stimstop'] * sample_rate)
            trial_data = pupil_diameter[start_sample:stop_sample]
            median_diameter = np.median(trial_data)
            medians.append(median_diameter)
        return np.array(medians)

    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)

    # Define range around the median to remove
    threshold = 0.04 * pup_median  # Adjust percentage 
    lower_bound = pup_median - threshold
    upper_bound = pup_median + threshold

    # Exclude trials near the median
    inds_low = np.where((pupil_trials < lower_bound))[0]  # Low trials excluding close-to-median
    inds_high = np.where((pupil_trials > upper_bound))[0]  # High trials excluding close-to-median

    print(f"Excluding trials in range: {lower_bound} - {upper_bound}")
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

        # Process data for low and high groups excluding close-to-median trials
        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[inds_high]['stimstart'], pre_time, post_time)

        baseline_low_firing_rate = np.mean(baseline_counts_low) / pre_time
        baseline_high_firing_rate = np.mean(baseline_counts_high) / pre_time
        evoked_low_firing_rate = np.mean(evoked_counts_low) / post_time
        evoked_high_firing_rate = np.mean(evoked_counts_high) / post_time

        # Fano Factor calculation
        FF_low = np.var(evoked_counts_low) / np.mean(evoked_counts_low) if np.mean(evoked_counts_low) > 0 else np.nan
        FF_high = np.var(evoked_counts_high) / np.mean(evoked_counts_high) if np.mean(evoked_counts_high) > 0 else np.nan

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
            'Fano Factor Low': FF_low,
            'Fano Factor High': FF_high
        })

    # Convert results to a DataFrame for this dataset and save
    results_df = pd.DataFrame(results)
    csv_filename = f'results_dataset_{dataset_num}.csv'
    csv_fullpath = os.path.join(results_dir, csv_filename)
    results_df.to_csv(csv_fullpath, index=False)

    all_results.append(results_df)

# Combine results across all datasets


combined_results_df = pd.concat(all_results, ignore_index=True)

# Create Population Bar Plot (aggregated across datasets)

# Calculate mean and standard error of evoked firing rates (Low vs. High)
mean_low = np.mean(combined_results_df['Evoked Low Firing Rate'])
mean_high = np.mean(combined_results_df['Evoked High Firing Rate'])
abs_firing = abs(mean_high-mean_low)


stderr_low = np.std(combined_results_df['Evoked Low Firing Rate']) / np.sqrt(len(combined_results_df))
stderr_high = np.std(combined_results_df['Evoked High Firing Rate']) / np.sqrt(len(combined_results_df))

# Paired t-test to compare Low vs. High evoked firing rates
_, p_val = ttest_rel(combined_results_df['Evoked Low Firing Rate'], combined_results_df['Evoked High Firing Rate'])

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar('Low', mean_low, yerr=stderr_low, color='gray', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_high, yerr=stderr_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)

# Annotate p-value
y_max = max(mean_low + stderr_low, mean_high + stderr_high)  # Maximum height of the bar + error
line_y = y_max + 2  # Adjust the height of the line

# Plot horizontal line and annotate p-value
plt.plot([0, 1], [line_y, line_y], color='black', lw=1.5)  # Horizontal line between bars
plt.text(0.5, line_y + 0.2, f'p = {p_val:.2e}', ha='center', fontsize=12, fontweight='bold')
plt.text(0.7 ,line_y + 0.3,"Low Pupil:2.38, High Pupil:2.54 mm", fontsize=10, fontweight='bold')
# Set labels and format the plot
plt.xlabel('Pupil Diameter', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Formatting the axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Save the bar plot
barplot_filename = 'population_barplot.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Create Fano Factor Scatter Plot: Fano Factor High vs Low
plt.figure(figsize=(8, 6))
plt.scatter(combined_results_df['Fano Factor Low'], combined_results_df['Fano Factor High'], color='green', s=15)
plt.plot([0.1, max(combined_results_df['Fano Factor Low'])],
         [0.1, max(combined_results_df['Fano Factor Low'])], 'r-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Fano Factor Low', fontsize=14)
plt.ylabel('Fano Factor High', fontsize=14)

# Set axis limits to start from 0.1
plt.xlim(0.1, max(combined_results_df['Fano Factor Low']))
plt.ylim(0.1, max(combined_results_df['Fano Factor High']))
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

# Save and display the scatter plot
scatterplot_filename = 'population_fano_factor_scatter_plot.svg'
plt.savefig(os.path.join(results_dir, scatterplot_filename))
plt.show()

# Bar Plot: Mean Fano Factor High vs Low
plt.figure(figsize=(8, 6))
mean_ff_low = np.nanmean(combined_results_df['Fano Factor Low'])
mean_ff_high = np.nanmean(combined_results_df['Fano Factor High'])
sem_ff_low = sem(combined_results_df['Fano Factor Low'], nan_policy='omit')
sem_ff_high = sem(combined_results_df['Fano Factor High'], nan_policy='omit')

plt.bar('Low', mean_ff_low, yerr=sem_ff_low, color='gray', capsize=5, edgecolor='black', linewidth=3)  # Increase the linewidth
plt.bar('High', mean_ff_high, yerr=sem_ff_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)  # Increase the linewidth

plt.xlabel('', fontsize=14)
plt.ylabel('Fano Factor', fontsize=14)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

# Save the bar plot
barplot_filename = 'population_mean_fano_factor_bar_plot.svg'
plt.savefig(os.path.join(results_dir, barplot_filename))
plt.show() 


# Create Population Scatter Plot (aggregated across datasets)

plt.figure(figsize=(8, 8))

low_population_means = combined_results_df['Evoked Low Firing Rate']
high_population_means = combined_results_df['Evoked High Firing Rate']


n = len(low_population_means)
# Create the scatter plot
plt.scatter(low_population_means, high_population_means, color='black', s=30, label=f'n = {n}')

# Add a diagonal reference line
max_limit = max(low_population_means.max(), high_population_means.max()) + 10
plt.plot([1, max_limit], [1, max_limit], 'r-', linewidth=2)

# Set the x and y scales to log
plt.xscale('log')
plt.yscale('log')

# Set the limits for the x and y axes
plt.xlim(1, max_limit)
plt.ylim(1, max_limit)

# Aspect ratio and axis formatting
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

# Set the labels and add penetration date to the scatter plot
plt.xlabel('Mean Firing Rate: Low', fontsize=16, fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=12, frameon=False)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Save and display the scatter plot
scatterplot_filename = 'Population_scatterplot.svg' 
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()



