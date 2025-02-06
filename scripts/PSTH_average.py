import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, pearsonr
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])
trial_metadata_path = utils.getFilePath(windowTitle="Metadata", filetypes=[('Mat-file', '*.mat')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)
mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
expt_info = mat['expt_info']

count_window = 20

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

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
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

all_mean_psth_low = []
all_mean_psth_high = []
# Define baseline period 
pre_time  = 0.15
post_time = 0.15
initial_time=0.035

# Extract stimuli from experiment info and add them to stimulusDF
def linear_stimuli(expt_info):
    stims = np.array([]) 
    for i in range(expt_info.trial_records.shape[0]):
        stims = np.hstack((stims, expt_info.trial_records[i].trImage))
    return stims

# Apply stimuli to stimulusDF
stimuli = linear_stimuli(expt_info)
stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]

#
stimulusDF = stimulusDF[stimulusDF['stimuli'] == 1]

all_mean_psth_low = []
all_mean_psth_high = []
# Define baseline period 
pre_time  = 0.15
post_time = 0.15
initial_time=0.035

# Process data for each cluster
for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
   
   
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15)
    
    baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time=0.15, post_time=0.15, initial_time=0.035)
    baseline_rate_mean = np.mean(baseline_rate_count) / 0.15
    evoked_rate_mean = np.mean(evoked_rate_count) / 0.15

    if evoked_rate_mean >= baseline_rate_mean + 5:
        valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
        valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

        spike_data_low = spike_data_all[valid_inds_low]
        spike_data_high = spike_data_all[valid_inds_high]
        
        mean_psth_low, _, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
        mean_psth_high, _, _, _, _ = meanvar_PSTH(spike_data_high, count_window)
        
        # Normalize the PSTHs
        max_psth_value = max(np.max(mean_psth_low), np.max(mean_psth_high))
        mean_psth_low /= max_psth_value
        mean_psth_high /= max_psth_value
        
        all_mean_psth_low.append(mean_psth_low)
        all_mean_psth_high.append(mean_psth_high)


avg_psth_low = np.mean(all_mean_psth_low, axis=0)
avg_psth_high = np.mean(all_mean_psth_high, axis=0)

# Calculate standard error
stderr_psth_low = np.std(all_mean_psth_low, axis=0) / np.sqrt(len(all_mean_psth_low))
stderr_psth_high = np.std(all_mean_psth_high, axis=0) / np.sqrt(len(all_mean_psth_high))


plt.figure(figsize=(10, 6))
time_vector = np.arange(-0.15, 0.15, 0.001) 
plt.plot(time_vector, avg_psth_low, color='black', linestyle='--', label='Low Arousal')
plt.plot(time_vector, avg_psth_high, color='red', label='High Arousal')

# Fill between for standard error
plt.fill_between(time_vector, avg_psth_low - stderr_psth_low, avg_psth_low + stderr_psth_low, color='grey', alpha=0.3)
plt.fill_between(time_vector, avg_psth_high - stderr_psth_high, avg_psth_high + stderr_psth_high, color='lightcoral', alpha=0.3)

plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)' , fontsize=16, fontweight='bold')
plt.ylim(0, 1)  
plt.xlim(-0.15, 0.15)  
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)   # Remove top spine
ax.spines['right'].set_visible(False) # Remove right spine
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(4)
avg_psth_filename = 'avg_psth_low_high_arousal_stim1.svg'
avg_psth_fullpath = os.path.join(results_dir, avg_psth_filename)
plt.savefig(avg_psth_fullpath)
plt.show()









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from utils import utils
from lib import readSGLX
import os

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

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

count_window = 20

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

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
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

all_mean_psth_low = []
all_mean_psth_high = []
# Define baseline period 
pre_time  = 0.15
post_time = 0.15
initial_time=0.035
# Process data for each cluster
for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
   
    # Extract spike data for trials
    spike_data_all = extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15)
    
    baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time=0.15, post_time=0.15, initial_time=0.035)
    baseline_rate_mean = np.mean(baseline_rate_count) / 0.15
    evoked_rate_mean = np.mean(evoked_rate_count) / 0.15

    if evoked_rate_mean >= baseline_rate_mean + 5:
        valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
        valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

        spike_data_low = spike_data_all[valid_inds_low]
        spike_data_high = spike_data_all[valid_inds_high]
        
        mean_psth_low, _, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
        mean_psth_high, _, _, _, _ = meanvar_PSTH(spike_data_high, count_window)
        
        # Normalize the PSTHs
        max_psth_value = max(np.max(mean_psth_low), np.max(mean_psth_high))
        mean_psth_low /= max_psth_value
        mean_psth_high /= max_psth_value
        
        all_mean_psth_low.append(mean_psth_low)
        all_mean_psth_high.append(mean_psth_high)

# Average PSTHs across all clusters
avg_psth_low = np.mean(all_mean_psth_low, axis=0)
avg_psth_high = np.mean(all_mean_psth_high, axis=0)

# Calculate standard error
stderr_psth_low = np.std(all_mean_psth_low, axis=0) / np.sqrt(len(all_mean_psth_low))
stderr_psth_high = np.std(all_mean_psth_high, axis=0) / np.sqrt(len(all_mean_psth_high))

# Plotting the averaged PSTH
plt.figure(figsize=(10, 6))
time_vector = np.arange(-0.15, 0.15, 0.001) 
plt.plot(time_vector, avg_psth_low, color='black', linestyle='--', label='Low Arousal')
plt.plot(time_vector, avg_psth_high, color='red', label='High Arousal')

# Fill between for standard error
plt.fill_between(time_vector, avg_psth_low - stderr_psth_low, avg_psth_low + stderr_psth_low, color='grey', alpha=0.3)
plt.fill_between(time_vector, avg_psth_high - stderr_psth_high, avg_psth_high + stderr_psth_high, color='lightcoral', alpha=0.3)

#plt.title('Low and High Arousal Pupil Diameter Effect')
plt.xlabel('Time (s)', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)' , fontsize=16, fontweight='bold')
plt.ylim(0, 1)  
plt.xlim(-0.15, 0.15)  
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)   # Remove top spine
ax.spines['right'].set_visible(False) # Remove right spine
ax.spines['bottom'].set_edgecolor('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_edgecolor('black')
ax.spines['left'].set_linewidth(4)
avg_psth_filename = 'avg_psth_low_high_arousal.svg'
avg_psth_fullpath = os.path.join(results_dir, avg_psth_filename)
plt.savefig(avg_psth_fullpath)
plt.show()






