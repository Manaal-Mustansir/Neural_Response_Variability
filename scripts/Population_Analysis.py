import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os
import seaborn as sns
from scipy import stats

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def linear_stimuli(expt_info):
    stims = np.array([])
    for i in range(expt_info.trial_records.shape[0]):
        stims = np.hstack((stims, expt_info.trial_records[i].trImage))
    return stims

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

all_results = []
threshold_value = 5

# Loop to load and process six datasets
for dataset_num in range(1, 7):
    print(f"Processing dataset {dataset_num}...")

    # Load data for the current dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", "*.npy")])

    # Load spike times and cluster data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)

    # Load the trial mat file and extract stimuli
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']
    stimuli = linear_stimuli(expt_info)
    
    # Load stimulus and trial DataFrames
    stimulusDF = pd.read_csv(stimulus_DF_path)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]
    trialDF = pd.read_csv(trial_DF_path)

    # Load metadata and pupil diameter data
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    # Calculate pupil diameter metrics
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Define baseline and evoked periods
    pre_time = 0.15
    post_time = 0.15
    initial_time = 0.035

    # Collect results for each cluster
    results = []

    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]
        
        # Process only Stimulus 1
        stim = 1  # Assuming Stimulus 1 is represented by the number 1 in your data
        stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == stim]

        # Ensure the indices are within the valid range for stimDF_tmp
        valid_inds_low = [i for i in inds_low if i < len(stimDF_tmp)]
        valid_inds_high = [i for i in inds_high if i < len(stimDF_tmp)]
    
        if len(valid_inds_low) == 0 or len(valid_inds_high) == 0:
            print(f"Skipping stimulus {stim} for cluster {c} in dataset {dataset_num} due to insufficient valid indices.")
            continue

        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_low]['stimstart'], pre_time, post_time, initial_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_high]['stimstart'], pre_time, post_time, initial_time)
        
        baseline_high_firing_rate = np.mean(baseline_counts_high) / pre_time
        baseline_low_firing_rate = np.mean(baseline_counts_low) / pre_time
        evoked_high_firing_rate = np.mean(evoked_counts_high) / post_time
        evoked_low_firing_rate = np.mean(evoked_counts_low) / post_time

        # Calculate Fano Factors
        FF_low = np.var(evoked_counts_low) / np.mean(evoked_counts_low) if np.mean(evoked_counts_low) > 0 else np.nan
        FF_high = np.var(evoked_counts_high) / np.mean(evoked_counts_high) if np.mean(evoked_counts_high) > 0 else np.nan

        # Apply the threshold condition
        if (evoked_high_firing_rate >= baseline_high_firing_rate + threshold_value) and (evoked_low_firing_rate >= baseline_low_firing_rate + threshold_value):

            AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

            t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
            t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

            baseline_class = 'no effect'
            evoked_class = 'no effect'
            if p_val_baseline < 0.05:
                baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
            if p_val_evoked < 0.05:
                evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

            results.append({
                'Cluster': c,
                'Stimulus': stim,
                'Penetration': f'Dataset {dataset_num}',
                'Monkey Name': 'Sansa',
                'Effect Size': AMI,
                'Baseline Classification': baseline_class,
                'Evoked Classification': evoked_class,
                'Baseline p-value': p_val_baseline,
                'Evoked p-value': p_val_evoked,
                'Baseline High Firing Rate': baseline_high_firing_rate,
                'Baseline Low Firing Rate': baseline_low_firing_rate,
                'Evoked High Firing Rate': evoked_high_firing_rate,
                'Evoked Low Firing Rate': evoked_low_firing_rate,
                'Fano Factor Low': FF_low,
                'Fano Factor High': FF_high
            })

    # Append results for the current dataset to the overall results
    all_results.extend(results)

# Convert all results to a DataFrame
all_results_df = pd.DataFrame(all_results)
csv_filename = 'stimulus_1_classification_results_with_fano_factor.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
all_results_df.to_csv(csv_fullpath, index=False)

# Scatter Plot: Fano Factor High vs Low
plt.figure(figsize=(8, 6))
plt.scatter(all_results_df['Fano Factor Low'], all_results_df['Fano Factor High'], color='blue', s=50)
plt.plot([0.1, max(all_results_df['Fano Factor Low'])],
         [0.1, max(all_results_df['Fano Factor Low'])], 'k--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Fano Factor Low', fontsize=14)
plt.ylabel('Fano Factor High', fontsize=14)
plt.title('Scatter Plot: Fano Factor (High vs Low)', fontsize=16)

# Set axis limits to start from 10^-1 (0.1)
plt.xlim(0.1, max(all_results_df['Fano Factor Low']))
plt.ylim(0.1, max(all_results_df['Fano Factor High']))

scatterplot_filename = 'fano_factor_scatter_plot.png'
plt.savefig(os.path.join(results_dir, scatterplot_filename))
plt.show()


# Bar Plot: Mean Fano Factor High vs Low
plt.figure(figsize=(8, 6))
mean_ff_low = np.nanmean(all_results_df['Fano Factor Low'])
mean_ff_high = np.nanmean(all_results_df['Fano Factor High'])
sem_ff_low = stats.sem(all_results_df['Fano Factor Low'], nan_policy='omit')
sem_ff_high = stats.sem(all_results_df['Fano Factor High'], nan_policy='omit')

plt.bar(['Low', 'High'], [mean_ff_low, mean_ff_high], yerr=[sem_ff_low, sem_ff_high], color=['grey', 'red'], capsize=5)
plt.xlabel('Condition', fontsize=14)
plt.ylabel('Mean Fano Factor', fontsize=14)
plt.title('Bar Plot: Mean Fano Factor (High vs Low)', fontsize=16)
barplot_filename = 'mean_fano_factor_bar_plot.png'
plt.savefig(os.path.join(results_dir, barplot_filename))
plt.show()

# Bar Plot: Evoked Firing Rates for Stimulus 1 (Original Plot)
plt.figure(figsize=(8, 6))
stim = 1  # Focus on Stimulus 1 only
stim_results = all_results_df[all_results_df['Stimulus'] == stim]

mean_low = np.mean(stim_results['Evoked Low Firing Rate'])
mean_high = np.mean(stim_results['Evoked High Firing Rate'])

stderr_low = np.std(stim_results['Evoked Low Firing Rate']) / np.sqrt(len(stim_results))
stderr_high = np.std(stim_results['Evoked High Firing Rate']) / np.sqrt(len(stim_results))

plt.bar('Low', mean_low, yerr=stderr_low, color='grey', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_high, yerr=stderr_high, color='red', capsize=5, edgecolor='black', linewidth=3, alpha=0.7)

plt.xlabel('Pupil Diameter', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.ylim(0, max(mean_low, mean_high) + 10)
plt.text(0.5, max(mean_low, mean_high) + 6, f'p = {p_val_evoked:.3e}', ha='center')

plt.title('Evoked Firing Rates for Stimulus 1', fontsize=18, fontweight='bold')
firingrate_barplot_filename = 'evoked_firing_rates_bar_plot.png'
plt.savefig(os.path.join(results_dir, firingrate_barplot_filename))
plt.show()

# Scatter Plot: High vs Low Firing Rates for Stimulus 1 (Original Plot)
plt.figure(figsize=(8, 6))
low_population_means = stim_results['Evoked Low Firing Rate']
high_population_means = stim_results['Evoked High Firing Rate']

plt.scatter(low_population_means, high_population_means, color='green', s=50, label='Stimulus 1')

max_limit = max(stim_results['Evoked Low Firing Rate'].max(), stim_results['Evoked High Firing Rate'].max()) + 10 
plt.plot([1, max_limit], [1, max_limit], 'k--', linewidth=2)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1, max_limit)
plt.ylim(1, max_limit)
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)
plt.xlabel('Mean Firing Rate: Low', fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontweight='bold')
plt.legend(loc='upper left', fontsize=12, frameon=False)

firingrate_scatterplot_filename = 'firing_rate_scatter_plot.png'
plt.savefig(os.path.join(results_dir, firingrate_scatterplot_filename))
plt.show()








