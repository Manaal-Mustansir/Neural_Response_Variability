
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, ttest_rel
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os
import seaborn as sns
import matplotlib.patches as mpatches

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"


# Constants for penetration date and monkey name
penetration_date = '2024-06-26'
monkey_name = 'WJ'

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_metadata_path = utils.getFilePath(windowTitle="Metadata", filetypes=[('Mat-file', '*.mat')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

# Load spike times and cluster data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)

# Load the trial mat file
mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
expt_info = mat['expt_info']

# extract stimuli from expt_info
def linear_stimuli(expt_info):
    stims = np.array([])
    for i in range(expt_info.trial_records.shape[0]):
        stims = np.hstack((stims, expt_info.trial_records[i].trImage))
    return stims

# Extract stimuli  and load  DataFrame
stimuli = linear_stimuli(expt_info)
stimulusDF = pd.read_csv(stimulus_DF_path)
stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]

# Load  data
trialDF = pd.read_csv(trial_DF_path)
binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

# Get spike times clustered 
spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, ax, stimulusDF, title, pre_time, post_time, color):
    tr = 0
    for _, row in stimulusDF.iterrows():
        tr += 1
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        ax.eventplot(spikes_in_trial - start_time, color=color, linewidths=0.5, lineoffsets=tr)

    # Set the y-ticks to be every 5 trials
    max_trial = len(stimulusDF)
    ax.set_yticks(np.arange(0, max_trial + 1, step=5))  # Gap of 5 between each tick
    ax.set_yticklabels(np.arange(0, max_trial + 1, step=5))  # Ensure labels match the ticks

    # Set title, labels, and font properties
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Trial', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.15, 0.15)

    # Make the x-axis tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')

    # Make y-axis tick labels bold
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')


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

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

count_window = 30
# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
pup_median = np.median(pupil_trials)
buffer = 0.05 * pup_median  # 5% buffer range
valid_trials = np.where((pupil_trials < pup_median - buffer) | (pupil_trials > pup_median + buffer))[0]

# Separate trials into low and high
inds_low = np.where(pupil_trials[valid_trials] < pup_median - buffer)[0]
inds_high = np.where(pupil_trials[valid_trials] > pup_median + buffer)[0]
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Store spike data for AMI calculation
all_spike_data_low = []
all_spike_data_high = []

# Define baseline period 
pre_time  = 0.15
post_time = 0.15
initial_time=0.035

def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

results = []



# Define constants for penetration and monkey name
penetration = '2024-06-26'
monkey_name = 'Wolfjaw'

for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    
    for stim in stimulusDF['stimuli'].unique():
        stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == stim]
    
        baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimDF_tmp["stimstart"].values, pre_time, post_time, initial_time)
        baseline_rate_mean = np.mean(baseline_rate_count) / pre_time
        evoked_rate_mean = np.mean(evoked_rate_count) / post_time

        if evoked_rate_mean >= baseline_rate_mean + 5:
            fig, axs = plt.subplots(3, 1, figsize=(10, 16))

            # Ensure the indices are within the valid range
            valid_inds_low = [i for i in inds_low if i < len(stimDF_tmp)]
            valid_inds_high = [i for i in inds_high if i < len(stimDF_tmp)]
            
            # Plotting Raster for Low Pupil Diameter
            plot_raster(Y, 
                        axs[0], 
                        stimDF_tmp.iloc[valid_inds_low], 
                        title=f'Cluster {c} - Stimulus {stim} (Low Pupil Diameter)',
                        pre_time=0.2,                    
                        post_time=0.2,
                        color='black')
            
            # Plotting Raster for High Pupil Diameter
            plot_raster(Y, 
                        axs[1], 
                        stimDF_tmp.iloc[valid_inds_high], 
                        title=f'Cluster {c} - Stimulus {stim} (High Pupil Diameter)',
                        pre_time=0.2,
                        post_time=0.2,
                        color='red')
            
            # Extracting Spike Data
            spike_data_all = extract_spike_data_for_trials(Y, stimDF_tmp, pre_time, post_time)

            # 
            valid_inds_low = [i for i in valid_inds_low if i < len(spike_data_all)]
            valid_inds_high = [i for i in valid_inds_high if i < len(spike_data_all)]

            spike_data_low = spike_data_all[valid_inds_low]
            spike_data_high = spike_data_all[valid_inds_high]
            
            # Continue with PSTH calculation and plotting
            mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
            mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window)

            total_spike_counts_low = np.sum(spike_data_low, axis=1)
            total_spike_counts_high = np.sum(spike_data_high, axis=1)
            
            baseline_low_firing_rate = np.mean(total_spike_counts_low)
            baseline_high_firing_rate = np.mean(total_spike_counts_high)

            fano_factor_low = var_psth_low / mean_psth_low
            fano_factor_high = var_psth_high / mean_psth_high

            bin_size = 0.001  # 1 ms bins
            time_vector = np.arange(-pre_time, post_time, bin_size)        

            stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
            stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

            normalizing_factor = count_window/1000.0

            # Plot firing rate PSTH
            axs[2].plot(time_vector, mean_psth_low / normalizing_factor, color='black', linestyle='--', label=f'Low Pupil Diameter - Stimulus {stim}')
            axs[2].plot(time_vector, mean_psth_high / normalizing_factor, color='red', label=f'High Pupil Diameter - Stimulus {stim}')

            axs[2].fill_between(time_vector, 
                    np.array(mean_psth_low - stderr_psth_low) / normalizing_factor, 
                    np.array(mean_psth_low + stderr_psth_low) / normalizing_factor, 
                    color='grey', alpha=0.3)
            axs[2].fill_between(time_vector, 
                    np.array(mean_psth_high - stderr_psth_high) / normalizing_factor, 
                    np.array(mean_psth_high + stderr_psth_high) / normalizing_factor, 
                    color='lightcoral', alpha=0.3)

            axs[2].set_xlim(-pre_time, post_time)

# Bold the x-axis and y-axis ticks for firing rate plot
            for label in axs[2].get_xticklabels():
                label.set_fontsize(12)
                label.set_fontweight('bold')

            for label in axs[2].get_yticklabels():  # This makes the firing rate ticks bold
                label.set_fontsize(12)
                label.set_fontweight('bold')

            axs[2].set_title(f'Cluster {c} - Stimulus {stim}', fontsize=16)
            axs[2].set_xlabel('Time (s)', fontsize=16, fontweight='bold')
            axs[2].set_ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')


            for ax in axs:
                ax.spines['left'].set_linewidth(4)    # Make left spine bold
                ax.spines['bottom'].set_linewidth(4)  # Make bottom spine bold
                ax.spines['top'].set_visible(False)   # Remove top spine
                ax.spines['right'].set_visible(False) # Remove right spine
                ax.tick_params(width=2)               # Make tick marks bold
            
            plt.tight_layout()
            svg_filename = f'cluster_{c}_stimulus_{stim}.svg'
            svg_fullpath = os.path.join(results_dir, svg_filename)
            plt.savefig(svg_fullpath)
            plt.close(fig)
            
            baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_low]['stimstart'], pre_time, post_time)
            baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_high]['stimstart'], pre_time, post_time)
            
            baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
            baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
            evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
            evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

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

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
csv_filename = 'classification_results_WJ.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
results_df.to_csv(csv_fullpath, index=False)

# Bar plot
results_df = results_df[results_df['Stimulus'] == 1]

# Bar plot 
plt.figure(figsize=(10, 6))

mean_low = np.mean(results_df['Evoked Low Firing Rate'])
mean_high = np.mean(results_df['Evoked High Firing Rate'])

stderr_low = np.std(results_df['Evoked Low Firing Rate']) / np.sqrt(len(results_df))
stderr_high = np.std(results_df['Evoked High Firing Rate']) / np.sqrt(len(results_df))

# Calculate p-value for paired t-test between Low and High firing rates
_, p_val = ttest_rel(results_df['Evoked Low Firing Rate'], results_df['Evoked High Firing Rate'])

# Create the bar plot
plt.bar('Low', mean_low, yerr=stderr_low, color='grey', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_high, yerr=stderr_high, color='red', capsize=5, edgecolor='black', linewidth=3, alpha=0.7)

# Draw the horizontal bar between the bars and annotate the p-value and penetration date
y_max = max(mean_low + stderr_low, mean_high + stderr_high)  # Maximum height of the bar + error
line_y = y_max + 2  # Adjust the height of the line

# Plot the horizontal line
plt.plot([0, 1], [line_y, line_y], color='black', lw=1.5)  # Horizontal line between bars

# Annotate the p-value and penetration date above the horizontal line
plt.text(0.5, line_y + 0.2, f'p = {p_val:.2e}', ha='center', fontsize=12, fontweight='bold')


# Set labels and format the plot
plt.xlabel('Pupil Diameter', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Formatting the axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Save and display the bar plot
barplot_filename = '2024-09-26_penetration_stimulus_1.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Scatter plot
plt.figure(figsize=(8, 8))

low_population_means = results_df['Evoked Low Firing Rate']
high_population_means = results_df['Evoked High Firing Rate']

# Calculate the number of data points "n"
n = len(low_population_means)

# Create the scatter plot
plt.scatter(low_population_means, high_population_means, color='black', s=30, label=f'n = {n}')

# Add a diagonal reference line
max_limit = max(results_df['Evoked Low Firing Rate'].max(), results_df['Evoked High Firing Rate'].max()) + 10 
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
plt.xlabel('Mean Firing Rate: Low', fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontweight='bold')
plt.text(1.2, max_limit * 0.9, f'Penetration: {penetration_date}', fontsize=12)

# Display "n" in the legend
plt.legend(loc='upper left', fontsize=12, frameon=False)

# Save and display the scatter plot
scatterplot_filename = '2024-09-26_penetration_scatter_plot.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()






















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, ttest_rel
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os
import seaborn as sns
import matplotlib.patches as mpatches

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_metadata_path = utils.getFilePath(windowTitle="Metadata", filetypes=[('Mat-file', '*.mat')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

# Load spike times and cluster data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)

# Load the trial mat file
mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
expt_info = mat['expt_info']

# extract stimuli from expt_info
def linear_stimuli(expt_info):
    stims = np.array([])
    for i in range(expt_info.trial_records.shape[0]):
        stims = np.hstack((stims, expt_info.trial_records[i].trImage))
    return stims

# Extract stimuli  and load  DataFrame
stimuli = linear_stimuli(expt_info)
stimulusDF = pd.read_csv(stimulus_DF_path)
stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]

# Load  data
trialDF = pd.read_csv(trial_DF_path)
binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

# Get spike times clustered 
spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, ax, stimulusDF, title, pre_time, post_time, color):
    tr = 0
    for _, row in stimulusDF.iterrows():
        tr += 1
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        ax.eventplot(spikes_in_trial - start_time, color=color, linewidths=0.5, lineoffsets=tr)
    
    # Set title, labels, and font properties
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Trial', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.15, 0.15)
    # Make the x-axis tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight('bold') 

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

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

count_window = 10
# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Store spike data for AMI calculation
all_spike_data_low = []
all_spike_data_high = []

# Define baseline period 
pre_time  = 0.15
post_time = 0.15
initial_time=0.035

def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

results = []



# Define constants for penetration and monkey name
penetration = '2024-04-17'
monkey_name = 'Sansa'

for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    
    for stim in stimulusDF['stimuli'].unique():
        stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == stim]
    
        baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimDF_tmp["stimstart"].values, pre_time, post_time, initial_time)
        baseline_rate_mean = np.mean(baseline_rate_count) / pre_time
        evoked_rate_mean = np.mean(evoked_rate_count) / post_time

        if evoked_rate_mean >= baseline_rate_mean + 5:
            fig, axs = plt.subplots(3, 1, figsize=(10, 16))

            # Ensure the indices are within the valid range
            valid_inds_low = [i for i in inds_low if i < len(stimDF_tmp)]
            valid_inds_high = [i for i in inds_high if i < len(stimDF_tmp)]
            
            # Plotting Raster for Low Pupil Diameter
            plot_raster(Y, 
                        axs[0], 
                        stimDF_tmp.iloc[valid_inds_low], 
                        title=f'Cluster {c} - Stimulus {stim} (Low Pupil Diameter)',
                        pre_time=0.2,                    
                        post_time=0.2,
                        color='black')
            
            # Plotting Raster for High Pupil Diameter
            plot_raster(Y, 
                        axs[1], 
                        stimDF_tmp.iloc[valid_inds_high], 
                        title=f'Cluster {c} - Stimulus {stim} (High Pupil Diameter)',
                        pre_time=0.2,
                        post_time=0.2,
                        color='red')
            
            # Extracting Spike Data
            spike_data_all = extract_spike_data_for_trials(Y, stimDF_tmp, pre_time, post_time)

            # 
            valid_inds_low = [i for i in valid_inds_low if i < len(spike_data_all)]
            valid_inds_high = [i for i in valid_inds_high if i < len(spike_data_all)]

            spike_data_low = spike_data_all[valid_inds_low]
            spike_data_high = spike_data_all[valid_inds_high]
            
            # Continue with PSTH calculation and plotting
            mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
            mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window)

            total_spike_counts_low = np.sum(spike_data_low, axis=1)
            total_spike_counts_high = np.sum(spike_data_high, axis=1)
            
            baseline_low_firing_rate = np.mean(total_spike_counts_low)
            baseline_high_firing_rate = np.mean(total_spike_counts_high)

            fano_factor_low = var_psth_low / mean_psth_low
            fano_factor_high = var_psth_high / mean_psth_high

            bin_size = 0.001  # 1 ms bins
            time_vector = np.arange(-pre_time, post_time, bin_size)        

            stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
            stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

            normalizing_factor = count_window/1000.0

            axs[2].plot(time_vector, mean_psth_low / normalizing_factor, color='black', linestyle='--', label=f'Low Pupil Diameter - Stimulus {stim}')
            axs[2].plot(time_vector, mean_psth_high / normalizing_factor, color='red', label=f'High Pupil Diameter - Stimulus {stim}')

            axs[2].fill_between(time_vector, 
                                np.array(mean_psth_low - stderr_psth_low) / normalizing_factor, 
                                np.array(mean_psth_low + stderr_psth_low) / normalizing_factor, 
                                color='grey', alpha=0.3)
            axs[2].fill_between(time_vector, 
                                np.array(mean_psth_high - stderr_psth_high) / normalizing_factor, 
                                np.array(mean_psth_high + stderr_psth_high) / normalizing_factor, 
                                color='lightcoral', alpha=0.3)

            axs[2].set_xlim(-pre_time, post_time)
            for label in axs[2].get_xticklabels():
                label.set_fontsize(18)
                label.set_fontweight('bold')
            axs[2].set_title(f'Cluster {c} - Stimulus {stim}', fontsize=16)
            axs[2].set_xlabel('Time (s)', fontsize=16, fontweight='bold')
            axs[2].set_ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')

            for ax in axs:
                ax.spines['left'].set_linewidth(4)    # Make left spine bold
                ax.spines['bottom'].set_linewidth(4)  # Make bottom spine bold
                ax.spines['top'].set_visible(False)   # Remove top spine
                ax.spines['right'].set_visible(False) # Remove right spine
                ax.tick_params(width=2)               # Make tick marks bold
            
            plt.tight_layout()
            svg_filename = f'cluster_{c}_stimulus_{stim}.svg'
            svg_fullpath = os.path.join(results_dir, svg_filename)
            plt.savefig(svg_fullpath)
            plt.close(fig)
            
            baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_low]['stimstart'], pre_time, post_time)
            baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_high]['stimstart'], pre_time, post_time)
            
            baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
            baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
            evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
            evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

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

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
csv_filename = 'classification_results_2024-04-17.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
results_df.to_csv(csv_fullpath, index=False)

# Visualization: Bar Plot for Population Effect with Stimuli Colors
plt.figure(figsize=(14, 8))

# Assigning colors for each stimulus
stimuli_colors = {stim: color for stim, color in zip(stimulusDF['stimuli'].unique(), sns.color_palette('hsv', len(stimulusDF['stimuli'].unique())))}

bar_width = 0.35
index = np.arange(len(stimulusDF['stimuli'].unique()))

# Plot bars for each stimulus
for i, stim in enumerate(stimulusDF['stimuli'].unique()):
    stim_results = results_df[results_df['Stimulus'] == stim]
    
    mean_low = np.mean(stim_results['Evoked Low Firing Rate'])
    mean_high = np.mean(stim_results['Evoked High Firing Rate'])
    
    stderr_low = np.std(stim_results['Evoked Low Firing Rate']) / np.sqrt(len(stim_results))
    stderr_high = np.std(stim_results['Evoked High Firing Rate']) / np.sqrt(len(stim_results))
    
    plt.bar(index[i] - bar_width/2, mean_low, bar_width, yerr=stderr_low, label=f'{stim} Low', color=stimuli_colors[stim])
    plt.bar(index[i] + bar_width/2, mean_high, bar_width, yerr=stderr_high, label=f'{stim} High', color=stimuli_colors[stim], alpha=0.7)

plt.xlabel('Stimulus', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.xticks(index, [f'Stimulus {stim}' for stim in stimulusDF['stimuli'].unique()], fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.legend(loc='upper right', fontsize=12, frameon=False)
plt.title('Evoked Firing Rates by Stimulus', fontsize=18, fontweight='bold')

barplot_filename = 'population_effect_by_stimulus.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Scatter Plot: High vs Low, with different colors for each stimulus
plt.figure(figsize=(8, 8))

for stim in stimulusDF['stimuli'].unique():
    stim_results = results_df[results_df['Stimulus'] == stim]
    low_population_means = stim_results['Evoked Low Firing Rate']
    high_population_means = stim_results['Evoked High Firing Rate']

    plt.scatter(low_population_means, high_population_means, color=stimuli_colors[stim], s=50, label=f'Stimulus {stim}')

# Add a diagonal line for reference
max_limit = max(results_df['Evoked Low Firing Rate'].max(), results_df['Evoked High Firing Rate'].max()) + 10 
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
plt.xlabel('Mean Firing Rate: Low' , fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontweight='bold')
plt.legend(loc='upper left', fontsize=12, frameon=False)
scatterplot_filename = 'high_vs_low_arousal_Scatter_plot_with_stimuli.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()

# Swarm Plot 
plt.figure(figsize=(10, 6))
sns.swarmplot(y="Evoked High Firing Rate", data=results_df, color='red')
sns.swarmplot(y="Evoked Low Firing Rate", data=results_df, color='black')
plt.xticks(['up', 'down', 'no effect'])
plt.yscale('log')
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='High Arousal')
black_patch = plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Low Arousal')
plt.legend(handles=[red_patch, black_patch])
plt.title('Evoked Firing Rates')
swarmplot_filename = 'swarm_plot.svg'
swarmplot_fullpath = os.path.join(results_dir, swarmplot_filename)
plt.savefig(swarmplot_fullpath)
plt.show()
