from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, ax, stimulusDF, title):
    for trial, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time - 0.15) & (Y <= stop_time)]
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=trial + 1)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')

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
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        if len(spikes_in_trial) == 0:
            continue
        spike_times_relative = spikes_in_trial - start_time
        hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def get_spike_counts(spike_times, stim_times, pre_time, post_time):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time) & (spike_times < stim_time + post_time)]
        
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

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
baseline_pre_time = 0.2

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
    
    # Calculate baseline spike rate
    baseline_rate = calculate_baseline_rate(Y, baseline_pre_time)
    
    # Skip clusters with baseline spike rate less than 1 Hz
    if baseline_rate < 1:
        continue
    
    # Create a single figure with subplots for rasters, PSTH, and Fano factor
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot Rasters 
    plot_raster(Y, axs[0], stimulusDF.iloc[inds_low], title=f'Cluster {c} - Low Pupil Diameter')
    plot_raster(Y, axs[1], stimulusDF.iloc[inds_high], title=f'Cluster {c} - High Pupil Diameter')
    
    # Extract spike data for all trials
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)

    if len(spike_data_all) == 0:
        continue

    # Ensure the indices 
    valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
    valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

    # Calculate PSTH for low and high pupil diameters 
    spike_data_low = spike_data_all[valid_inds_low]
    spike_data_high = spike_data_all[valid_inds_high]
    
    mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window=20)
    mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window=20)

    # Calculate Fano factor
    fano_factor_low = var_psth_low / mean_psth_low
    fano_factor_high = var_psth_high / mean_psth_high

    # Ensure time_vector matches PSTH data length
    bin_size = 0.001  # 1 ms bins
    time_vector = np.arange(-0.15, 0.15, bin_size)

    # Skip bad clusters with NaNs or Infs
    if np.any(np.isnan(mean_psth_low)) or np.any(np.isnan(mean_psth_high)) or np.any(np.isinf(mean_psth_low)) or np.any(np.isinf(mean_psth_high)):
        continue

    if np.any(np.isnan(fano_factor_low)) or np.any(np.isnan(fano_factor_high)) or np.any(np.isinf(fano_factor_low)) or np.any(np.isinf(fano_factor_high)):
        continue

    # Calculate standard error
    stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
    stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

    # Plot PSTH and Fano factor for low and high pupil diameters
    axs[2].plot(time_vector, mean_psth_low * 10, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[2].plot(time_vector, mean_psth_high * 10, color='red', label='High Pupil Diameter')

    # Add fill_between for PSTH low
    axs[2].fill_between(time_vector, (mean_psth_low - stderr_psth_low) * 10, (mean_psth_low + stderr_psth_low) * 10, color='grey', alpha=0.3)
    # Add fill_between for PSTH high
    axs[2].fill_between(time_vector, (mean_psth_high - stderr_psth_high) * 10, (mean_psth_high + stderr_psth_high) * 10, color='lightcoral', alpha=0.3)

    axs[2].set_xlim(-0.15, 0.15)  
    axs[2].set_title(f'PSTH for Cluster {c}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Spike Rate (Hz)')
    axs[2].legend()

    axs[3].plot(time_vector, fano_factor_low, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[3].plot(time_vector, fano_factor_high, color='red', label='High Pupil Diameter')
    axs[3].set_xlim(-0.15, 0.15) 
    axs[3].set_title(f'Fano Factor for Cluster {c}')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Fano Factor')
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(f'cluster_{c}.svg')
    plt.close(fig)

    # Calculate spike counts for low and high pupil states
    baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], baseline_pre_time, 0.2)
    baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], baseline_pre_time, 0.2)

    # Calculate mean firing rates for each condition
    baseline_high_firing_rate = np.mean(baseline_counts_high)
    baseline_low_firing_rate = np.mean(baseline_counts_low)
    evoked_high_firing_rate = np.mean(evoked_counts_high)
    evoked_low_firing_rate = np.mean(evoked_counts_low)

    # Calculate AMI
    AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

    # Perform t-tests for baseline and evoked counts
    t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
    t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

    # Classify based on p-values
    baseline_class = 'no effect'
    evoked_class = 'no effect'
    if p_val_baseline < 0.05:
        baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
    if p_val_evoked < 0.05:
        evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

    # Append results to the list
    results.append({
        'Cluster': c,
        'Penetration': penetration,
        'Monkey Name': monkey_name,
        'Effect Size': AMI,  # Store the AMI here
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

# Save the DataFrame as a CSV file
results_df.to_csv('classification_results_2024-04-17.csv', index=False)

# Aggregate PSTH data for each category
psth_categories = {
    'Baseline Up - Evoked Down': [],
    'Baseline Up - Evoked Up': [],
    'Baseline Down - Evoked Up': [],
    'Baseline Down - Evoked Down': []
}

for result in results:
    baseline_class = result['Baseline Classification']
    evoked_class = result['Evoked Classification']
    
    if baseline_class == 'up' and evoked_class == 'down':
        psth_categories['Baseline Up - Evoked Down'].append(result)
    elif baseline_class == 'up' and evoked_class == 'up':
        psth_categories['Baseline Up - Evoked Up'].append(result)
    elif baseline_class == 'down' and evoked_class == 'up':
        psth_categories['Baseline Down - Evoked Up'].append(result)
    elif baseline_class == 'down' and evoked_class == 'down':
        psth_categories['Baseline Down - Evoked Down'].append(result)

# Function to calculate the average PSTH for a list of results
def calculate_average_psth(results, time_vector, count_window):
    psth_sum = np.zeros(len(time_vector))
    count = 0
    
    for result in results:
        cluster = result['Cluster']
        Y = spike_times_clusters[cluster]
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)
        mean_psth, _, _, _, _ = meanvar_PSTH(spike_data_all, count_window=count_window)
        psth_sum += mean_psth
        count += 1
    
    return psth_sum / count if count > 0 else None

# Plot each category's PSTHs with their average
colors = {
    'Baseline Up - Evoked Down': 'blue',
    'Baseline Up - Evoked Up': 'green',
    'Baseline Down - Evoked Up': 'orange',
    'Baseline Down - Evoked Down': 'purple'
}

count_window = 20
bin_size = 0.001
time_vector = np.arange(-0.15, 0.15, bin_size)

for category, results in psth_categories.items():
    plt.figure(figsize=(10, 6))
    
    # Plot individual PSTHs
    for result in results:
        cluster = result['Cluster']
        Y = spike_times_clusters[cluster]
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)
        mean_psth, _, _, _, _ = meanvar_PSTH(spike_data_all, count_window=count_window)
        plt.plot(time_vector, mean_psth * 10, color=colors[category], alpha=0.3)
    
    # Calculate and plot the average PSTH
    avg_psth = calculate_average_psth(results, time_vector, count_window)
    if avg_psth is not None:
        plt.plot(time_vector, avg_psth * 10, color=colors[category], linewidth=2, label=f'Average {category}')
    
    plt.title(f'PSTHs - {category}')
    plt.xlabel('Time (s)')
    plt.ylabel('Spike Rate (Hz)')
    plt.legend()
    plt.savefig(f'psth_category_{category.replace(" ", "_").replace("-", "").lower()}.svg')
    plt.close()
from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, ax, stimulusDF, title):
    for trial, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time - 0.15) & (Y <= stop_time)]
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=trial + 1)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')

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
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        if len(spikes_in_trial) == 0:
            continue
        spike_times_relative = spikes_in_trial - start_time
        hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def get_spike_counts(spike_times, stim_times, pre_time, post_time):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time) & (spike_times < stim_time + post_time)]
        
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

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
baseline_pre_time = 0.2

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
    
    # Calculate baseline spike rate
    baseline_rate = calculate_baseline_rate(Y, baseline_pre_time)
    
    # Skip clusters with baseline spike rate less than 1 Hz
    if baseline_rate < 1:
        continue
    
    # Create a single figure with subplots for rasters, PSTH, and Fano factor
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot Rasters 
    plot_raster(Y, axs[0], stimulusDF.iloc[inds_low], title=f'Cluster {c} - Low Pupil Diameter')
    plot_raster(Y, axs[1], stimulusDF.iloc[inds_high], title=f'Cluster {c} - High Pupil Diameter')
    
    # Extract spike data for all trials
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)

    if len(spike_data_all) == 0:
        continue

    # Ensure the indices 
    valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
    valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

    # Calculate PSTH for low and high pupil diameters 
    spike_data_low = spike_data_all[valid_inds_low]
    spike_data_high = spike_data_all[valid_inds_high]
    
    mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window=20)
    mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window=20)

    # Calculate Fano factor
    fano_factor_low = var_psth_low / mean_psth_low
    fano_factor_high = var_psth_high / mean_psth_high

    # Ensure time_vector matches PSTH data length
    bin_size = 0.001  # 1 ms bins
    time_vector = np.arange(-0.15, 0.15, bin_size)

    # Skip bad clusters with NaNs or Infs
    if np.any(np.isnan(mean_psth_low)) or np.any(np.isnan(mean_psth_high)) or np.any(np.isinf(mean_psth_low)) or np.any(np.isinf(mean_psth_high)):
        continue

    if np.any(np.isnan(fano_factor_low)) or np.any(np.isnan(fano_factor_high)) or np.any(np.isinf(fano_factor_low)) or np.any(np.isinf(fano_factor_high)):
        continue

    # Calculate standard error
    stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
    stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

    # Plot PSTH and Fano factor for low and high pupil diameters
    axs[2].plot(time_vector, mean_psth_low * 10, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[2].plot(time_vector, mean_psth_high * 10, color='red', label='High Pupil Diameter')

    # Add fill_between for PSTH low
    axs[2].fill_between(time_vector, (mean_psth_low - stderr_psth_low) * 10, (mean_psth_low + stderr_psth_low) * 10, color='grey', alpha=0.3)
    # Add fill_between for PSTH high
    axs[2].fill_between(time_vector, (mean_psth_high - stderr_psth_high) * 10, (mean_psth_high + stderr_psth_high) * 10, color='lightcoral', alpha=0.3)

    axs[2].set_xlim(-0.15, 0.15)  
    axs[2].set_title(f'PSTH for Cluster {c}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Spike Rate (Hz)')
    axs[2].legend()

    axs[3].plot(time_vector, fano_factor_low, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[3].plot(time_vector, fano_factor_high, color='red', label='High Pupil Diameter')
    axs[3].set_xlim(-0.15, 0.15) 
    axs[3].set_title(f'Fano Factor for Cluster {c}')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Fano Factor')
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(f'cluster_{c}.svg')
    plt.close(fig)

    # Calculate spike counts for low and high pupil states
    baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], baseline_pre_time, 0.2)
    baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], baseline_pre_time, 0.2)

    # Calculate mean firing rates for each condition
    baseline_high_firing_rate = np.mean(baseline_counts_high)
    baseline_low_firing_rate = np.mean(baseline_counts_low)
    evoked_high_firing_rate = np.mean(evoked_counts_high)
    evoked_low_firing_rate = np.mean(evoked_counts_low)

    # Calculate AMI
    AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

    # Perform t-tests for baseline and evoked counts
    t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
    t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

    # Classify based on p-values
    baseline_class = 'no effect'
    evoked_class = 'no effect'
    if p_val_baseline < 0.05:
        baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
    if p_val_evoked < 0.05:
        evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

    # Append results to the list
    results.append({
        'Cluster': c,
        'Penetration': penetration,
        'Monkey Name': monkey_name,
        'Effect Size': AMI,  # Store the AMI here
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

# Save the DataFrame as a CSV file
results_df.to_csv('classification_results_2024-04-17.csv', index=False)

# Bar plot for population effect with added categories
categories = ['Baseline Up - Evoked Down', 'Baseline Up - Evoked Up', 
              'Baseline Down - Evoked Up', 'Baseline Down - Evoked Down']

category_counts = {category: 0 for category in categories}

for result in results:
    baseline_class = result['Baseline Classification']
    evoked_class = result['Evoked Classification']
    
    if baseline_class == 'up' and evoked_class == 'down':
        category_counts['Baseline Up - Evoked Down'] += 1
    elif baseline_class == 'up' and evoked_class == 'up':
        category_counts['Baseline Up - Evoked Up'] += 1
    elif baseline_class == 'down' and evoked_class == 'up':
        category_counts['Baseline Down - Evoked Up'] += 1
    elif baseline_class == 'down' and evoked_class == 'down':
        category_counts['Baseline Down - Evoked Down'] += 1

# Plot proportions
total_counts = sum(category_counts.values())
proportions = [count / total_counts for count in category_counts.values()]

plt.figure(figsize=(10, 6))
plt.bar(categories, proportions, color=['blue', 'green', 'orange', 'purple'])
plt.title('Proportions of PSTH Categories')
plt.ylabel('Proportion')
plt.show()

# Scatter plot for high vs low population
plt.figure(figsize=(10, 6))
low_population_means = [res['Baseline Low Firing Rate'] for res in results]
high_population_means = [res['Baseline High Firing Rate'] for res in results]
plt.scatter(low_population_means, high_population_means, color='purple')
plt.plot([min(low_population_means), max(low_population_means)], [min(low_population_means), max(low_population_means)], 'k--')
plt.title('High vs Low Arousal')
plt.xlabel('Low Population Mean Firing Rate')
plt.ylabel('High Population Mean Firing Rate')
plt.show()

# Aggregate PSTH data for each category
psth_categories = {
    'Baseline Up - Evoked Down': [],
    'Baseline Up - Evoked Up': [],
    'Baseline Down - Evoked Up': [],
    'Baseline Down - Evoked Down': []
}

for result in results:
    baseline_class = result['Baseline Classification']
    evoked_class = result['Evoked Classification']
    
    if baseline_class == 'up' and evoked_class == 'down':
        psth_categories['Baseline Up - Evoked Down'].append(result)
    elif baseline_class == 'up' and evoked_class == 'up':
        psth_categories['Baseline Up - Evoked Up'].append(result)
    elif baseline_class == 'down' and evoked_class == 'up':
        psth_categories['Baseline Down - Evoked Up'].append(result)
    elif baseline_class == 'down' and evoked_class == 'down':
        psth_categories['Baseline Down - Evoked Down'].append(result)

# Function to calculate the average PSTH for a list of results
def calculate_average_psth(results, time_vector, count_window):
    psth_sum = np.zeros(len(time_vector))
    count = 0
    
    for result in results:
        cluster = result['Cluster']
        Y = spike_times_clusters[cluster]
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)
        mean_psth, _, _, _, _ = meanvar_PSTH(spike_data_all, count_window=count_window)
        psth_sum += mean_psth
        count += 1
    
    return psth_sum / count if count > 0 else None

# Plot each category's average PSTH
colors = {
    'Baseline Up - Evoked Down': 'blue',
    'Baseline Up - Evoked Up': 'green',
    'Baseline Down - Evoked Up': 'orange',
    'Baseline Down - Evoked Down': 'purple'
}

count_window = 20
bin_size = 0.001
time_vector = np.arange(-0.15, 0.15, bin_size)

for category, results in psth_categories.items():
    avg_psth = calculate_average_psth(results, time_vector, count_window)
    
    if avg_psth is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(time_vector, avg_psth * 10, color=colors[category], label=category)
        plt.title(f'Average PSTH - {category}')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike Rate (Hz)')
        plt.legend()
        plt.savefig(f'average_psth_{category.replace(" ", "_").replace("-", "").lower()}.svg')
        plt.close()


from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, ax, stimulusDF, title):
    for trial, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time - 0.15) & (Y <= stop_time)]
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=trial + 1)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')

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
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        if len(spikes_in_trial) == 0:
            continue
        spike_times_relative = spikes_in_trial - start_time
        hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def get_spike_counts(spike_times, stim_times, pre_time, post_time):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time) & (spike_times < stim_time + post_time)]
        
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

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
baseline_pre_time = 0.2

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
    
    # Calculate baseline spike rate
    baseline_rate = calculate_baseline_rate(Y, baseline_pre_time)
    
    # Skip clusters with baseline spike rate less than 1 Hz
    if baseline_rate < 1:
        continue
    
    # Create a single figure with subplots for rasters, PSTH, and Fano factor
    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot Rasters 
    plot_raster(Y, axs[0], stimulusDF.iloc[inds_low], title=f'Cluster {c} - Low Pupil Diameter')
    plot_raster(Y, axs[1], stimulusDF.iloc[inds_high], title=f'Cluster {c} - High Pupil Diameter')
    
    # Extract spike data for all trials
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)

    if len(spike_data_all) == 0:
        continue

    # Ensure the indices 
    valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
    valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

    # Calculate PSTH for low and high pupil diameters 
    spike_data_low = spike_data_all[valid_inds_low]
    spike_data_high = spike_data_all[valid_inds_high]
    
    mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window=20)
    mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window=20)

    # Calculate Fano factor
    fano_factor_low = var_psth_low / mean_psth_low
    fano_factor_high = var_psth_high / mean_psth_high

    # Ensure time_vector matches PSTH data length
    bin_size = 0.001  # 1 ms bins
    time_vector = np.arange(-0.15, 0.15, bin_size)

    # Skip bad clusters with NaNs or Infs
    if np.any(np.isnan(mean_psth_low)) or np.any(np.isnan(mean_psth_high)) or np.any(np.isinf(mean_psth_low)) or np.any(np.isinf(mean_psth_high)):
        continue

    if np.any(np.isnan(fano_factor_low)) or np.any(np.isnan(fano_factor_high)) or np.any(np.isinf(fano_factor_low)) or np.any(np.isinf(fano_factor_high)):
        continue

    # Calculate standard error
    stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
    stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

    # Plot PSTH and Fano factor for low and high pupil diameters
    axs[2].plot(time_vector, mean_psth_low * 10, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[2].plot(time_vector, mean_psth_high * 10, color='red', label='High Pupil Diameter')

    # Add fill_between for PSTH low
    axs[2].fill_between(time_vector, (mean_psth_low - stderr_psth_low) * 10, (mean_psth_low + stderr_psth_low) * 10, color='grey', alpha=0.3)
    # Add fill_between for PSTH high
    axs[2].fill_between(time_vector, (mean_psth_high - stderr_psth_high) * 10, (mean_psth_high + stderr_psth_high) * 10, color='lightcoral', alpha=0.3)

    axs[2].set_xlim(-0.15, 0.15)  
    axs[2].set_title(f'PSTH for Cluster {c}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Spike Rate (Hz)')
    axs[2].legend()

    axs[3].plot(time_vector, fano_factor_low, color='black', linestyle='--', label='Low Pupil Diameter')
    axs[3].plot(time_vector, fano_factor_high, color='red', label='High Pupil Diameter')
    axs[3].set_xlim(-0.15, 0.15) 
    axs[3].set_title(f'Fano Factor for Cluster {c}')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Fano Factor')
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(f'cluster_{c}.svg')
    plt.close(fig)

    # Calculate spike counts for low and high pupil states
    baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], baseline_pre_time, 0.2)
    baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], baseline_pre_time, 0.2)

    # Calculate mean firing rates for each condition
    baseline_high_firing_rate = np.mean(baseline_counts_high)
    baseline_low_firing_rate = np.mean(baseline_counts_low)
    evoked_high_firing_rate = np.mean(evoked_counts_high)
    evoked_low_firing_rate = np.mean(evoked_counts_low)

    # Calculate AMI
    AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

    # Perform t-tests for baseline and evoked counts
    t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
    t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

    # Classify based on p-values
    baseline_class = 'no effect'
    evoked_class = 'no effect'
    if p_val_baseline < 0.05:
        baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
    if p_val_evoked < 0.05:
        evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

    # Append results to the list
    results.append({
        'Cluster': c,
        'Penetration': penetration,
        'Monkey Name': monkey_name,
        'Effect Size': AMI,  # Store the AMI here
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

# Save the DataFrame as a CSV file
results_df.to_csv('classification_results_2024-04-17.csv', index=False)

# Bar plot for population effect with added categories
categories = ['Baseline Up - Evoked Down', 'Baseline Up - Evoked Up', 
              'Baseline Down - Evoked Up', 'Baseline Down - Evoked Down']

category_counts = {category: 0 for category in categories}

for result in results:
    baseline_class = result['Baseline Classification']
    evoked_class = result['Evoked Classification']
    
    if baseline_class == 'up' and evoked_class == 'down':
        category_counts['Baseline Up - Evoked Down'] += 1
    elif baseline_class == 'up' and evoked_class == 'up':
        category_counts['Baseline Up - Evoked Up'] += 1
    elif baseline_class == 'down' and evoked_class == 'up':
        category_counts['Baseline Down - Evoked Up'] += 1
    elif baseline_class == 'down' and evoked_class == 'down':
        category_counts['Baseline Down - Evoked Down'] += 1

# Plot proportions
total_counts = sum(category_counts.values())
proportions = [count / total_counts for count in category_counts.values()]

plt.figure(figsize=(10, 6))
plt.bar(categories, proportions, color=['blue', 'green', 'orange', 'purple'])
plt.title('Proportions of PSTH Categories')
plt.ylabel('Proportion')
plt.show()

# Scatter plot for high vs low population
plt.figure(figsize=(10, 6))
low_population_means = [res['Baseline Low Firing Rate'] for res in results]
high_population_means = [res['Baseline High Firing Rate'] for res in results]
plt.scatter(low_population_means, high_population_means, color='purple')
plt.plot([min(low_population_means), max(low_population_means)], [min(low_population_means), max(low_population_means)], 'k--')
plt.title('High vs Low Arousal')
plt.xlabel('Low Population Mean Firing Rate')
plt.ylabel('High Population Mean Firing Rate')
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from utils import utils
from lib import readSGLX

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15, sRate=30000, bin_size=0.001):
    trial_spike_data = []
    bin_edges = np.arange(-pre_time, post_time + bin_size, bin_size)
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        if len(spikes_in_trial) == 0:
            continue
        spike_times_relative = spikes_in_trial - start_time
        hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]


for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    
    # Extract spike data for all trials
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)

    if len(spike_data_all) == 0:
        continue

    # Ensure indices are within bounds
    valid_inds_low = inds_low[inds_low < len(spike_data_all)]
    valid_inds_high = inds_high[inds_high < len(spike_data_all)]


    # Calculate PSTH for low and high pupil diameters using meanvar_PSTH function
    spike_data_low = spike_data_all[valid_inds_low]
    spike_data_high = spike_data_all[valid_inds_high]
    
    mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low)
    mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high)

    # Calculate Fano factor
    fano_factor_low = var_psth_low / mean_psth_low
    fano_factor_high = var_psth_high / mean_psth_high

    # Ensure time_vector matches PSTH data length
    bin_size = 0.001  # 1 ms bins
    time_vector = np.arange(-0.15, 0.15, bin_size)

    # Skip bad clusters with NaNs or Infs
    if np.any(np.isnan(mean_psth_low)) or np.any(np.isnan(mean_psth_high)) or np.any(np.isinf(mean_psth_low)) or np.any(np.isinf(mean_psth_high)):
       
        continue

    if np.any(np.isnan(fano_factor_low)) or np.any(np.isnan(fano_factor_high)) or np.any(np.isinf(fano_factor_low)) or np.any(np.isinf(fano_factor_high)):
       
        continue

    # Plot PSTH and Fano factor for low and high pupil diameters
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot PSTH
    ax1.plot(time_vector, mean_psth_low, color='black', linestyle='--', label='Low Pupil Diameter')
    ax1.plot(time_vector, mean_psth_high, color='red', label='High Pupil Diameter')
    ax1.set_title(f'PSTH for Cluster {c}')
    ax1.set_xlabel('Time (s)')
    ax1.legend()

    # Plot Fano factor
    ax2.plot(time_vector, fano_factor_low, color='black', linestyle='--', label='Low Pupil Diameter')
    ax2.plot(time_vector, fano_factor_high, color='red', label='High Pupil Diameter')
    ax2.set_title(f'Fano Factor for Cluster {c}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Fano Factor')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()










