import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from utils import utils
from lib import readSGLX


# Function to calculate median pupil diameter
def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

# Function to extract spike data for each trial
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

# Function to calculate mean and variance of PSTH
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

# Function to calculate baseline spike rate
def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

# Function to load and process each dataset
def process_dataset(spike_times_secpath, clusters_path, stimulus_DF_path, trial_DF_path, binFullPath, pupilFullPath):
    # Load data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    trialDF = pd.read_csv(trial_DF_path)

    # Read meta data and sampling rate for this dataset
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Accumulate all PSTH data
    all_psth_low = []
    all_psth_high = []

    # Define baseline period 
    baseline_pre_time = 0.2

    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]
        
        #Calculate baseline spike rate
        baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time=0.15, post_time=0.15)
        baseline_rate_mean= np.mean(baseline_rate_count)/0.15
        evoked_rate_mean = np.mean(evoked_rate_count)/0.15

        if evoked_rate_mean >=  baseline_rate_mean + 10:
            
            
            # Extract spike data for all trials
            spike_data_all = extract_spike_data_for_trials(Y, stimulusDF)
            if len(spike_data_all) == 0:
                continue

            # Ensure the indices are valid
            valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
            valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

            # Calculate PSTH for low and high pupil diameters
            spike_data_low = spike_data_all[valid_inds_low]
            spike_data_high = spike_data_all[valid_inds_high]
            
            mean_psth_low, _, _, _, _ = meanvar_PSTH(spike_data_low, count_window=20)
            mean_psth_high, _, _, _, _ = meanvar_PSTH(spike_data_high, count_window=20)
            
            # Normalize the PSTHs by their respective maxima for each cluster
            max_low = np.max(mean_psth_low)
            max_high = np.max(mean_psth_high)

                #max low or max high which one is high
            
            max_low_high = np.max([max_low, max_high])

        
            if max_low_high > 0:
                mean_psth_low_normalized = mean_psth_low / max_low_high
                all_psth_low.append(mean_psth_low_normalized)
            else:
                print(f"Cluster {c}: max_low is 0, skipping normalization for low arousal.")

            if max_low_high > 0:
                mean_psth_high_normalized = mean_psth_high / max_low_high
                all_psth_high.append(mean_psth_high_normalized)
            else:
                print(f"Cluster {c}: max_high is 0, skipping normalization for high arousal.")

        # Convert lists to arrays for easier averaging
        all_psth_low = np.array(all_psth_low)
        all_psth_high = np.array(all_psth_high)

        return all_psth_low, all_psth_high

# Load datasets
datasets = []
for i in range(6):
    print(f"Processing dataset {i+1}...")
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {i+1}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {i+1}", filetypes=[('Clusters numpy file', '*.npy')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {i+1}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {i+1}", filetypes=[('trial csv file', '*.csv')])
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {i+1}", filetypes=[("NIdq binary", "*.bin")])
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {i+1}", filetypes=[("Numpy array", "*.npy")])
    
    psth_low, psth_high = process_dataset(spike_times_secpath, clusters_path, stimulus_DF_path, trial_DF_path, binFullPath, pupilFullPath)
    datasets.append((psth_low, psth_high))

# Accumulate PSTH data across all datasets
accumulated_psth_low = []
accumulated_psth_high = []

for psth_low, psth_high in datasets:
    accumulated_psth_low.extend(psth_low)
    accumulated_psth_high.extend(psth_high)

accumulated_psth_low = np.array(accumulated_psth_low)
accumulated_psth_high = np.array(accumulated_psth_high)

# Time vector should match the length of PSTH data
bin_size = 0.001  # 1 ms bins
time_vector = np.arange(-0.15, 0.15, bin_size)

# Calculate the average and standard error of the normalized PSTH across all datasets
average_norm_psth_low = np.mean(accumulated_psth_low, axis=0)
stderr_norm_psth_low = np.std(accumulated_psth_low, axis=0) / np.sqrt(accumulated_psth_low.shape[0])

average_norm_psth_high = np.mean(accumulated_psth_high, axis=0) if len(accumulated_psth_high) > 0 else np.array([])
stderr_norm_psth_high = np.std(accumulated_psth_high, axis=0) / np.sqrt(accumulated_psth_high.shape[0]) if len(accumulated_psth_high) > 0 else np.array([])

# Plot average normalized PSTH with error bars across all datasets
plt.figure(figsize=(12, 8))

# Plot the average normalized PSTH with error shading
plt.plot(time_vector, average_norm_psth_low, color='blue', label='Low Arousal', linewidth=3)
plt.fill_between(time_vector, average_norm_psth_low - stderr_norm_psth_low, average_norm_psth_low  + stderr_norm_psth_low , color='blue', alpha=0.3)

if len(average_norm_psth_high) > 0:
    plt.plot(time_vector, average_norm_psth_high, color='red', label='High Arousal', linewidth=3)
    plt.fill_between(time_vector, average_norm_psth_high  - stderr_norm_psth_high , average_norm_psth_high + stderr_norm_psth_high , color='red', alpha=0.3)

# Enhance the title, labels, and add grid
plt.title('Average PSTH for Low and High Arousal Across All Datasets', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Normalized Spike Rate', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig('average_psth_all_datasets_with_error.svg')
plt.show()
















