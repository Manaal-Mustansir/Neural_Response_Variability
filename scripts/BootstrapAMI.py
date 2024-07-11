

from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats
from utils import utils as ut
from scipy.signal import detrend

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = ut.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = ut.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, cluster_index, ax, stimulusDF, title):
    for trial, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time - 0.15) & (Y <= stop_time)]
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=trial + 1)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_xlim(-0.15, 0.15)

def calculate_mean_firing_rate(Y, trialDF, i, yesplot=False, pdfname=None):
    spike_counts = []  
    if yesplot:
        fig, ax = plt.subplots()
    for index, row in trialDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time + 0.03) & (Y <= stop_time)]
        spike_count_trial = len(spikes_in_trial)
        spike_counts.append(spike_count_trial)
        baseline_start = start_time - 0.2
        baseline_end = start_time
        trialDF.at[index, f'baseline_spike_count_cluster_{i}'] = len(spikes_in_trial)
        trialDF.at[index, f'baseline_firing_rate_cluster_{i}'] = len(spikes_in_trial) / (baseline_end - baseline_start) if baseline_end > baseline_start else 0
    if yesplot:
        plt.show()
    return trialDF

def get_spike_counts(Y, stimulusDF):
    spkC_evk = np.NaN * np.ones(len(stimulusDF))
    spkC_bsl = np.NaN * np.ones(len(stimulusDF))
    for index, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time + 0.035) & (Y <= stop_time)]
        spikes_in_bsl = Y[(Y >= start_time - 0.15 + 0.035) & (Y <= start_time)]
        spkC_evk[index] = len(spikes_in_trial)
        spkC_bsl[index] = len(spikes_in_bsl)
    return spkC_evk, spkC_bsl

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

spkC = {}
BSL = {}
spkC_zscored = {}
good_clusters = []
for i in spike_times_clusters.keys():
    Y = spike_times_clusters[i]
    spkC[i], BSL[i] = get_spike_counts(Y, stimulusDF)
    res = stats.ttest_rel(spkC[i], BSL[i])
    if np.mean(spkC[i]) > np.mean(BSL[i]) + 3:
        good_clusters.append(i)
        spkC_zscored[i] = stats.zscore(spkC[i])

pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

spkC_low = []
spkC_high = []
pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]
for c in good_clusters:
    spkC_low.append(spkC[c][inds_low])
    spkC_high.append(spkC[c][inds_high])

spkC_low = np.array(spkC_low)
spkC_high = np.array(spkC_high)

#  (AMI)
ami = np.mean(spkC_high) / np.mean(spkC_low)
print(f"Arousal Modulation Index (AMI): {ami}")

# bootstrapping 
def bootstrap_ami(spkC_low, spkC_high, n_bootstrap=10000):
    ami_bootstrap = []
    combined_data = np.concatenate([spkC_low.flatten(), spkC_high.flatten()])
    for _ in range(n_bootstrap):
        low_sample = np.random.choice(combined_data, size=spkC_low.size, replace=True)
        high_sample = np.random.choice(combined_data, size=spkC_high.size, replace=True)
        ami_bootstrap.append(np.mean(high_sample) / np.mean(low_sample))
    return ami_bootstrap

ami_bootstrap = bootstrap_ami(spkC_low, spkC_high)
p_value = np.sum(np.array(ami_bootstrap) >= ami) / len(ami_bootstrap)
print(f"Bootstrap p-value: {p_value}")

# Plot AMI 
spkC_low = spkC_low / (0.15 - 0.035)
mnLow = np.mean(spkC_low)
spkC_high = spkC_high / (0.15 - 0.035)
mnHig = np.mean(spkC_high)
SELow = np.std(spkC_low) / np.sqrt(len(spkC_low.flatten()))
SEHig = np.std(spkC_high) / np.sqrt(len(spkC_high.flatten()))

fig, ax = plt.subplots()
ax.bar([1, 2], [mnLow, mnHig], yerr=[SELow, SEHig])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low arousal', 'High arousal'])
ax.set_title(f'Arousal Modulation Index: {ami:.2f}, p-value: {p_value:.4f}')
plt.show()

# rasters for low and high arousal states
for c in good_clusters:
    Y = spike_times_clusters[c]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plot_raster(Y, c, axs[0], stimulusDF.iloc[inds_low], title='Low arousal')
    plot_raster(Y, c, axs[1], stimulusDF.iloc[inds_high], title='High arousal')
    plt.tight_layout()
    plt.show()

#  rasters by pupil diameter

for c in good_clusters:
    Y = spike_times_clusters[c]
    sorted_indices = np.argsort(pupil_trials)
    sorted_stimulusDF = stimulusDF.iloc[sorted_indices]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    plot_raster(Y, c, axs[0], sorted_stimulusDF.iloc[inds_low], title='Rasters for Low Pupil Diameter')
    plot_raster(Y, c, axs[1], sorted_stimulusDF.iloc[inds_high], title='Rasters for High Pupil Diameter')
    plt.tight_layout()
    plt.show()




# Variability calculation
variability_low = np.std(spkC_low, axis=1) / np.mean(spkC_low, axis=1)
variability_high = np.std(spkC_high, axis=1) / np.mean(spkC_high, axis=1)

# Plotting the variability
fig, ax = plt.subplots()
ax.bar([1, 2], [np.mean(variability_low), np.mean(variability_high)], yerr=[np.std(variability_low), np.std(variability_high)])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low arousal', 'High arousal'])
ax.set_title('Variability of Neuronal Responses: Low vs High Arousal')
plt.show()