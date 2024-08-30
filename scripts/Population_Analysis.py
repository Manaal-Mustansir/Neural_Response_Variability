import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, ttest_rel
from utils import utils
from lib import readSGLX
import os
import seaborn as sns
import matplotlib.patches as mpatches

# results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Define a list to store results from all datasets
all_results = []

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15, bin_size=0.001):
    trial_spike_data = []
    bin_edges = np.arange(-pre_time, post_time + bin_size, bin_size)
    expected_hist_length = len(bin_edges) - 1
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        
        if len(spikes_in_trial) == 0:
            hist = np.zeros(expected_hist_length)
        else:
            spike_times_relative = spikes_in_trial - start_time
            hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000, binarize: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if binarize:
        data = data > 0
    
    if style == 'valid':
        mean_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        tmp = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan * np.ones((data.shape[1]))
        vari_timevector = np.nan * np.ones((data.shape[1]))
        tmp = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        tmp[i,:] = np.convolve(data[i,:], np.ones(count_window,), style)
            
    vari_timevector = np.var(tmp, axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0], (tmp.shape[0], nboots))
        mean_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:], axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:], axis=0)
    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted


# Loop to load and process six datasets
all_psth_low = []
all_psth_high = []

for dataset_num in range(1, 7):
    print(f"Processing dataset {dataset_num}...")

    # Load data for the current dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", "*.npy")])

    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    trialDF = pd.read_csv(trial_DF_path)
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupil_diameter = np.load(pupilFullPath)
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    # Pupil diameter processing
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    results = []

    # Define constants for penetration and monkey name
    penetration = f'2024-04-17 Dataset {dataset_num}'
    monkey_name = 'Sansa'
    pre_time  = 0.15
    post_time = 0.15
    initial_time=0.035

    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]

        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], pre_time, post_time, initial_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], pre_time, post_time, initial_time)

        baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
        baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
        evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
        evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

        # Apply the threshold condition
        if evoked_high_firing_rate >= baseline_high_firing_rate + 5 and evoked_low_firing_rate >= baseline_low_firing_rate + 5:
            
            AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

            t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
            t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

            baseline_class = 'no effect'
            evoked_class = 'no effect'
            if p_val_baseline < 0.05:
                baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
            if p_val_evoked < 0.05:
                evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

            # Calculate PSTH for this cluster
            spike_data_all = extract_spike_data_for_trials(Y, stimulusDF, pre_time, post_time)
            valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
            valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

            spike_data_low = spike_data_all[valid_inds_low]
            spike_data_high = spike_data_all[valid_inds_high]
            
            mean_psth_low, _, _, _, _ = meanvar_PSTH(spike_data_low, count_window=20)
            mean_psth_high, _, _, _, _ = meanvar_PSTH(spike_data_high, count_window=20)
            
            all_psth_low.append(mean_psth_low)
            all_psth_high.append(mean_psth_high)

            results.append({
                'Cluster': c,
                'Penetration': penetration,
                'Monkey Name': monkey_name,
                'Effect Size': AMI,
                'Layers': '',
                'Baseline Classification': baseline_class,
                'Evoked Classification': evoked_class,
                'Baseline p-value': p_val_baseline,
                'Evoked p-value': p_val_evoked,
                'Baseline High Firing Rate': baseline_high_firing_rate,
                'Baseline Low Firing Rate': baseline_low_firing_rate,
                'Evoked High Firing Rate': evoked_high_firing_rate,
                'Evoked Low Firing Rate': evoked_low_firing_rate
            })

    # Append the results of the current dataset to the overall results
    all_results.extend(results)

# After processing all datasets, convert the combined results to a DataFrame
all_results_df = pd.DataFrame(all_results)
csv_filename = 'combined_classification_results.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
all_results_df.to_csv(csv_fullpath, index=False)

# Calculate the mean PSTH across all clusters and datasets
population_psth_low = np.mean(all_psth_low, axis=0)
population_psth_high = np.mean(all_psth_high, axis=0)

# Calculate the standard error for the PSTH
stderr_psth_low = np.std(all_psth_low, axis=0) / np.sqrt(len(all_psth_low))
stderr_psth_high = np.std(all_psth_high, axis=0) / np.sqrt(len(all_psth_high))

# Create time vector for PSTH plot
bin_size = 0.001  # 1 ms bins
time_vector = np.arange(-pre_time, post_time, bin_size)

# Plot population PSTH
plt.figure(figsize=(10, 6))
plt.plot(time_vector, population_psth_low, color='black', label='Low Arousal')
plt.fill_between(time_vector, population_psth_low - stderr_psth_low, population_psth_low + stderr_psth_low, color='black', alpha=0.3)
plt.plot(time_vector, population_psth_high, color='red', label='High Arousal')
plt.fill_between(time_vector, population_psth_high - stderr_psth_high, population_psth_high + stderr_psth_high, color='red', alpha=0.3)

plt.xlim(-pre_time, post_time)
plt.title('Low and High Pupil Diameter Effect')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)  # Make bold
psth_filename = 'population_psth_with_error.svg'
psth_fullpath = os.path.join(results_dir, psth_filename)
plt.savefig(psth_fullpath)
plt.show()

# Bar plot for population effect
mean_low_population = np.mean([res['Evoked Low Firing Rate'] for res in all_results])
mean_high_population = np.mean([res['Evoked High Firing Rate'] for res in all_results])
n_neurons = len([res['Evoked High Firing Rate'] for res in all_results])

# Calculate p-value for the bar plot
_, p_val = ttest_rel([res['Evoked Low Firing Rate'] for res in all_results],
                     [res['Evoked High Firing Rate'] for res in all_results])

# Ensure p_val is not rounded to zero
p_val_str = f'{p_val:.3e}' if p_val < 0.001 else f'{p_val:.3f}'

# Standard errors
stderr_low_population = np.std([res['Evoked Low Firing Rate'] for res in all_results]) / np.sqrt(len(all_results))
stderr_high_population = np.std([res['Evoked High Firing Rate'] for res in all_results]) / np.sqrt(len(all_results))

plt.figure(figsize=(10, 6))

#  matplotlib text 
labels = ['Low', 'High']
colors = ['grey', 'red']
bars = plt.bar(labels, 
               [mean_low_population, mean_high_population], 
               yerr=[stderr_low_population, stderr_high_population], 
               color=colors,  
               capsize=5,
               edgecolor='black',
               linewidth=3)

# Add markers on top of the bars
for bar, color in zip(bars, colors):
    plt.scatter(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                color=color, zorder=3)  # Adjust 's' for marker size

plt.ylim(0, max(mean_low_population, mean_high_population) + 10)
plt.text(0.5, max(mean_low_population, mean_high_population) + 6, f'p = {p_val_str}', ha='center')
plt.plot([0, 1], [max(mean_low_population, mean_high_population) + 5] * 2, color='black')

plt.xlabel('Pupil Diameter')
plt.ylabel('Mean Firing Rate (Hz)')
ax = plt.gca()
ax.spines['top'].set_visible(False)   # Remove top spine
ax.spines['right'].set_visible(False) # Remove right spine
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(2)
barplot_filename = 'population_effect_combined.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Scatter plot for high vs low across all datasets
plt.figure(figsize=(6, 6))
low_population_means = [res['Evoked Low Firing Rate'] for res in all_results]
high_population_means = [res['Evoked High Firing Rate'] for res in all_results]

# Determine the limit based on the maximum value across both axes, but allowing a margin to include all points
max_limit = max(max(low_population_means), max(high_population_means)) + 20  
plt.scatter(low_population_means, high_population_means, color='black', s=8)  # Set 's' marker size
plt.plot([1, max_limit], [1, max_limit], 'r-', linewidth=2)  #  middle diagonal

# Set the plot box (spine) color 
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(2)
plt.xscale('log')
plt.yscale('log')

plt.xlim(1, max_limit)
plt.ylim(1, max_limit)
plt.gca().set_aspect('equal', adjustable='box')
#plt.title('Low and High Pupil Diameter Effect')
plt.xlabel('Mean Firing Rate (Hz): Low Pupil Diameter')
plt.ylabel('Mean Firing Rate (Hz): High Pupil Diameter')
plt.legend(loc='upper left')
scatterplot_filename = 'high_vs_low_arousal_scatter_combined.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()


# Swarm plot across all datasets
plt.figure(figsize=(10, 6))
sns.swarmplot(x="Evoked Classification", y="Evoked High Firing Rate", data=all_results_df, color='red')
sns.swarmplot(x="Evoked Classification", y="Evoked Low Firing Rate", data=all_results_df, color='black')
plt.xticks(['up', 'down', 'no effect'])
plt.yscale('log')
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='High Arousal')
black_patch = plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Low Arousal')
plt.legend(handles=[red_patch, black_patch])
plt.title('Low and High Pupil Diameter Effect')
swarmplot_filename = 'swarm_plot_combined.svg'
swarmplot_fullpath = os.path.join(results_dir, swarmplot_filename)
plt.savefig(swarmplot_fullpath)
plt.show()



