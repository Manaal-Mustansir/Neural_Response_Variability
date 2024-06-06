import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils  
from lib import readSGLX
from scipy.signal import butter, filtfilt, iirnotch

# Notch filter
def apply_notch_filter(LFP_avg, fs, freq=60.0, quality_factor=40.0):
    b_notch, a_notch = iirnotch(freq, quality_factor, fs)
    filtered_data = filtfilt(b_notch, a_notch, LFP_avg, axis=1)
    return filtered_data

# Butterworth bandpass filter
def apply_bandpass_filter(LFP_avg, fs, lowcut=1.0, highcut=100.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, LFP_avg, axis=1)
    return filtered_data

# Baseline correction
def apply_baseline_correction(LFP_avg, pre_stim_samples):
    baseline = np.mean(LFP_avg[:, :pre_stim_samples], axis=1, keepdims=True)
    return LFP_avg - baseline

# Function to convert 2D LFP data to 1D by averaging every two consecutive channels
def convert_to_1d(LFP_avg):
    nchannels = LFP_avg.shape[0]
    LFP_1d = np.zeros((nchannels // 2, LFP_avg.shape[1]))
    for i in range(0, nchannels - 1, 2):
        LFP_1d[i//2, :] = (LFP_avg[i, :] + LFP_avg[i + 1, :]) / 2
    return LFP_1d

# Calculate CSD
def calculate_csd(LFP_avg_1d, dx=1):
    dLFP = np.gradient(LFP_avg_1d, axis=0) / dx
    d2LFP = np.gradient(dLFP, axis=0) / dx
    return d2LFP

# Calculate z-score
def calculate_zscore(CSD, baseline_samples):
    mean = np.mean(CSD[:, baseline_samples:], axis=1, keepdims=True)
    std = np.std(CSD[:, baseline_samples:], axis=1, keepdims=True)
    zscore = (CSD - mean) / std
    return zscore

# File paths
binFullPath_meta = utils.getFilePath(windowTitle="Select meta lf file", filetypes=[("sGLX lf", "*.meta")])
binFullPath_bin = utils.getFilePath(windowTitle="Select binary lf file", filetypes=[("sGLX lf", "*.bin")])
stimulus_DF_path = utils.getFilePath(windowTitle="Select stimulus CSV file", filetypes=[('stimulus csv file', '*.csv')])

meta = readSGLX.readMeta(binFullPath_meta)
sRate = readSGLX.SampRate(meta)

pre_stim_time = 0.15
post_stim_time = 0.15
pre_stim_samples = int(pre_stim_time * sRate)
post_stim_samples = int(post_stim_time * sRate)

rawData = readSGLX.makeMemMapRaw(binFullPath_bin, meta)

stimulusDF = pd.read_csv(stimulus_DF_path)
stimstart_times = stimulusDF['stimstart'].values

nStimuli = len(stimstart_times)
nchannels = rawData.shape[0]
nsamples = pre_stim_samples + post_stim_samples

LFP = np.nan * np.ones((nchannels, nsamples, nStimuli))

for i, stimstart in enumerate(stimstart_times):
    start_sample = int((stimstart - pre_stim_time) * sRate)
    end_sample = int((stimstart + post_stim_time) * sRate)
    
    if start_sample >= 0 and end_sample < rawData.shape[1]:
        segment = rawData[:, start_sample:end_sample]
        LFP[:, :, i] = segment

LFP_avg = np.nanmean(LFP, axis=2)

# Apply notch filter
LFP_avg_notch = apply_notch_filter(LFP_avg, sRate)

# Apply Butterworth bandpass filter
LFP_avg_filtered = apply_bandpass_filter(LFP_avg_notch, sRate, lowcut=1.0, highcut=100.0)

# Apply baseline correction
LFP_avg_filtered = apply_baseline_correction(LFP_avg_filtered, pre_stim_samples)

# Convert LFP_avg to 1D by averaging every two consecutive channels
LFP_avg_1d = convert_to_1d(LFP_avg_filtered)
LFP_avg_raw_1d = convert_to_1d(LFP_avg)

CSD = calculate_csd(LFP_avg_1d)
CSD_raw = calculate_csd(LFP_avg_raw_1d)

# Calculate z-score of CSD using only post-stimulus samples for z-score calculation
CSD_zscore = calculate_zscore(CSD, pre_stim_samples)
CSD_raw_zscore = calculate_zscore(CSD_raw, pre_stim_samples)

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(20, 20))

# Plot LFPs
offset = 5  
scale = 8  
time_axis = np.linspace(-pre_stim_time, post_stim_time, nsamples) * 1000  # Time axis in ms
axs[0, 0].set_title('Filtered LFP Averages with Notch and Butterworth Filters')
for ch in range(nchannels):
    axs[0, 0].plot(time_axis, LFP_avg_filtered[ch, :] * scale + ch * offset, color='black')
axs[0, 0].axvline(x=0, color='blue', linestyle='-')
axs[0, 0].axvline(x=post_stim_time * 1000, color='magenta', linestyle='-')
axs[0, 0].set_xlabel('Time (ms)')

# Plot Filtered CSD
im = axs[0, 1].imshow(CSD, aspect='auto', origin='lower', cmap='RdBu_r', extent=[-pre_stim_time * 1000, post_stim_time * 1000, 0, CSD.shape[0]])
fig.colorbar(im, ax=axs[0, 1], label='Source/Sink')
axs[0, 1].axvline(x=0, color='blue', linestyle='-')
axs[0, 1].axvline(x=post_stim_time * 1000, color='magenta', linestyle='-')
axs[0, 1].set_xlabel('Time from stimulus onset (ms)')
axs[0, 1].set_title('Current Source Density (CSD) - Filtered Data')

# Plot Raw CSD
im = axs[1, 0].imshow(CSD_raw, aspect='auto', origin='lower', cmap='RdBu_r', extent=[-pre_stim_time * 1000, post_stim_time * 1000, 0, CSD_raw.shape[0]])
fig.colorbar(im, ax=axs[1, 0], label='Source/Sink')
axs[1, 0].axvline(x=0, color='blue', linestyle='-')
axs[1, 0].axvline(x=post_stim_time * 1000, color='magenta', linestyle='-')
axs[1, 0].set_xlabel('Time from stimulus onset (ms)')
axs[1, 0].set_title('Current Source Density (CSD) - Raw Data')

# Plot Z-score of Raw CSD
im = axs[1, 1].imshow(CSD_raw_zscore, aspect='auto', origin='lower', cmap='RdBu_r', extent=[-pre_stim_time * 1000, post_stim_time * 1000, 0, CSD_raw_zscore.shape[0]])
fig.colorbar(im, ax=axs[1, 1], label='Source/Sink')
axs[1, 1].axvline(x=0, color='blue', linestyle='-')
axs[1, 1].axvline(x=post_stim_time * 1000, color='magenta', linestyle='-')
axs[1, 1].set_xlabel('Time from stimulus onset (ms)')
axs[1, 1].set_title('Z-score of Current Source Density (CSD) - Raw Data')

# Plot Z-score of Filtered CSD
im = axs[2, 0].imshow(CSD_zscore, aspect='auto', origin='lower', cmap='RdBu_r', extent=[-pre_stim_time * 1000, post_stim_time * 1000, 0, CSD_zscore.shape[0]])
fig.colorbar(im, ax=axs[2, 0], label='Source/Sink')
axs[2, 0].axvline(x=0, color='blue', linestyle='-')
axs[2, 0].axvline(x=post_stim_time * 1000, color='magenta', linestyle='-')
axs[2, 0].set_xlabel('Time from stimulus onset (ms)')
axs[2, 0].set_title('Z-score of Current Source Density (CSD) - Filtered Data')

# Remove the empty plot (bottom right)
fig.delaxes(axs[2, 1])

plt.tight_layout()
plt.show()




