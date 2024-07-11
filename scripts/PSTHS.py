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
    ax1.set_ylabel('Spike Rate (Hz)')
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












from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import detrend

# Load files
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_DF_path = utils.getFilePath(windowTitle="trials_DF_Stimulus", filetypes=[('Trials_Stimulus csv file', '*.csv')])
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data",filetypes=[("Numpy array","*.npy")])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter = np.load(pupilFullPath)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file",filetypes=[("NIdq binary","*.bin")]) # type: ignore
meta  = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

# Get spike times for each cluster
spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def meanvar_PSTH(data, count_window=100, style='same', return_bootdstrs=False, nboots=1000):
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

def return_psth(Y, trialDF, bin_size=0.001, pre_window=0.150, post_window=0.150):
    num_bins = int((pre_window + post_window) / bin_size)
    data_matrix = np.zeros((len(trialDF), num_bins))

    for trial_idx, row in trialDF.iterrows():
        stimstart = row['stimstart']

        start_time = stimstart - pre_window
        stop_time = stimstart + post_window

        spikes_in_window = Y[(Y >= start_time) & (Y <= stop_time)]
        bin_counts, _ = np.histogram(spikes_in_window, bins=np.linspace(start_time, stop_time, num_bins + 1))
        data_matrix[trial_idx, :] = bin_counts

    mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted = meanvar_PSTH(data_matrix, count_window=100, style='same', return_bootdstrs=True, nboots=1000)
    t = np.linspace(-pre_window, post_window, num_bins)
    return mean_timevector, vari_timevector, mean_timevector_booted, vari_timevector_booted, t

def get_spike_counts(Y, stimulusDF):
    spkC_evk = np.NaN * np.ones(len(stimulusDF))
    spkC_bsl = np.NaN * np.ones(len(stimulusDF))
                    
    for index, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time + 0.035) & (Y <= stop_time)]
        spikes_in_bsl = Y[(Y >= start_time - 0.15 + 0.035) & (Y <= start_time)]
        spike_count_trial = len(spikes_in_trial)
        spike_count_bsl = len(spikes_in_bsl)
        spkC_evk[index] = spike_count_trial
        spkC_bsl[index] = spike_count_bsl

    return spkC_evk, spkC_bsl

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample  = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def get_pupil_Diameter(stimulusDF, pupil_diameter, sRate):
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    return inds_low, inds_high

good_clusters = []
spkC = {}
BSL = {}
spkC_zscored = {}

inds_low, inds_high = get_pupil_Diameter(trialDF, pupil_diameter, sRate)

mean_timevectors_all = []
vari_timevectors_all = []
fano_factors_all = []

for i in spike_times_clusters.keys():
    Y = spike_times_clusters[i]
    spkC[i], BSL[i] = get_spike_counts(Y, trialDF)    
    if np.mean(spkC[i]) > np.mean(BSL[i]) + 3:
        good_clusters.append(i)
        
        mean_timevector, vari_timevector, mean_timevector_booted, vari_timevector_booted, t = return_psth(Y, trialDF)
        fano_factor = vari_timevector / mean_timevector

        mean_timevectors_all.append(mean_timevector)
        vari_timevectors_all.append(vari_timevector)
        fano_factors_all.append(fano_factor)
        
        fig, ax = plt.subplots(2, 1)
        psth_low = mean_timevector - np.std(mean_timevector_booted, axis=0)
        psth_high = mean_timevector + np.std(mean_timevector_booted, axis=0)
        
        ax[0].fill_between(t, psth_low, psth_high, color='grey')
        ax[0].plot(t, mean_timevector, 'k--')      
        ax[1].plot(t, fano_factor, 'r-')
        plt.show()

# Compute population 
mean_timevectors_all = np.array(mean_timevectors_all)
vari_timevectors_all = np.array(vari_timevectors_all)
fano_factors_all = np.array(fano_factors_all)

population_mean_timevector = np.mean(mean_timevectors_all, axis=0)
population_vari_timevector = np.mean(vari_timevectors_all, axis=0)
population_fano_factor = np.mean(fano_factors_all, axis=0)

fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Population PSTH
ax[0].plot(t, population_mean_timevector, 'k--', label='Population-2024-05-31 ')
ax[0].fill_between(t, population_mean_timevector - np.std(mean_timevectors_all, axis=0), population_mean_timevector + np.std(mean_timevectors_all, axis=0), color='grey', alpha=0.5)
ax[0].set_title('Population ')
ax[0].set_xlabel('Time (s)')
ax[0].legend()

# Population variance
ax[1].plot(t, population_vari_timevector, 'r-', label='Population Variance-2024-05-')
ax[1].set_title('Population Variance')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Variance')
ax[1].legend()

# Population Fano factor
ax[2].plot(t, population_fano_factor, 'b-', label='Population Fano Factor')
ax[2].set_title('Population Fano Factor')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Fano Factor')
ax[2].legend()

plt.tight_layout()
plt.show()
