import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from utils import utils as ut
from lib import readSGLX
import os

# Constants
pre_time = 0.15  # Pre-stimulus window (-0.15s)
post_time = 0.15  # Post-stimulus window (+0.15s)
chanList = [2]   
num_datasets = 6  # Number of datasets to process

# Function to calculate median pupil diameter for each trial based on stimulus windows
def calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sample_rate):
    medians = []
    for trial_start, trial_stop in zip(trial_starts, trial_stops):
        trial_medians = []
        for stim_start, stim_stop in zip(stim_starts, stim_stops):
            if stim_start >= trial_start and stim_stop <= trial_stop:
                start_sample = int((stim_start - pre_time) * sample_rate)
                stop_sample = int((stim_stop + post_time) * sample_rate)
                trial_data = pupil_diameter[start_sample:stop_sample]
                median_diameter = np.median(trial_data)
                trial_medians.append(median_diameter)
        if trial_medians:
            medians.append(np.median(trial_medians))
    return np.array(medians)

# Initialize lists to hold averages for all datasets
all_mean_eye_trace_low = []
all_mean_eye_trace_high = []
all_stderr_eye_trace_low = []
all_stderr_eye_trace_high = []

# Loop over 6 datasets
for i in range(num_datasets):
    print(f"Processing dataset {i+1}/{num_datasets}...")
    
    # Load binary file and metadata
    binFullPath = ut.getFilePath(windowTitle=f"Binary nidq file {i+1}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)

    # Load pupil diameter data
    pupilFullPath = ut.getFilePath(windowTitle=f"Pupil diameter data {i+1}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)

    # Load trial start/stop times from trial DataFrame
    trial_DF_path = ut.getFilePath(windowTitle="Trial CSV", filetypes=[("CSV file", "*.csv")])
    trialDF = pd.read_csv(trial_DF_path)

    # Load stimulus start/stop times from stimulus DataFrame
    stimulusDF_path = ut.getFilePath(windowTitle="Stimulus CSV", filetypes=[("CSV file", "*.csv")])
    stimulusDF = pd.read_csv(stimulusDF_path)

    # Use the correct column names 'trialstart' and 'trialstop' from trialDF
    trial_starts = trialDF['trialstart'].values
    trial_stops = trialDF['trialstop'].values

    # Use 'stimstart' and 'stimstop' from stimulusDF
    stim_starts = stimulusDF['stimstart'].values
    stim_stops = stimulusDF['stimstop'].values

    # Calculate median pupil diameter for each trial
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sRate)

    # Split into low and high pupil diameter trials
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Trial-based pupil trace extraction (-0.15 to +0.15s around stimulus onset)
    trial_eye_traces = []
    time_window_samples = int((pre_time + post_time) * sRate)  # Total samples in window

    for stim_start in stim_starts:
        stim_index = int(stim_start * sRate)
        pre_stim_index = stim_index - int(pre_time * sRate)
        post_stim_index = stim_index + int(post_time * sRate)
        
        trial_eye_trace = np.full(time_window_samples, np.nan)
        valid_start = max(0, pre_stim_index)
        valid_end = min(len(pupil_diameter), post_stim_index)
        insert_start = max(0, -pre_stim_index)
        insert_end = insert_start + (valid_end - valid_start)
        
        trial_eye_trace[insert_start:insert_end] = pupil_diameter[valid_start:valid_end]
        trial_eye_traces.append(trial_eye_trace)

    # Convert to NumPy array
    trial_eye_traces = np.array(trial_eye_traces)

    # Separate into low and high pupil diameter trials
    low_pupil_traces = trial_eye_traces[inds_low]
    high_pupil_traces = trial_eye_traces[inds_high]

    # Compute average pupil diameter across low and high trials
    mean_eye_trace_low = np.nanmean(low_pupil_traces, axis=0)
    mean_eye_trace_high = np.nanmean(high_pupil_traces, axis=0)

    # Compute standard error for low and high trials
    stderr_eye_trace_low = np.nanstd(low_pupil_traces, axis=0) / np.sqrt(len(low_pupil_traces))
    stderr_eye_trace_high = np.nanstd(high_pupil_traces, axis=0) / np.sqrt(len(high_pupil_traces))

    # Store results for final average calculation across datasets
    all_mean_eye_trace_low.append(mean_eye_trace_low)
    all_mean_eye_trace_high.append(mean_eye_trace_high)
    all_stderr_eye_trace_low.append(stderr_eye_trace_low)
    all_stderr_eye_trace_high.append(stderr_eye_trace_high)

# Compute average across datasets
mean_eye_trace_low_avg = np.mean(all_mean_eye_trace_low, axis=0)
mean_eye_trace_high_avg = np.mean(all_mean_eye_trace_high, axis=0)
stderr_eye_trace_low_avg = np.mean(all_stderr_eye_trace_low, axis=0)
stderr_eye_trace_high_avg = np.mean(all_stderr_eye_trace_high, axis=0)

# Time vector from -0.15 to +0.15s
time_vector = np.linspace(-pre_time, post_time, len(mean_eye_trace_low_avg))

# Plot the average pupil diameter trace for low and high conditions across all datasets
plt.figure(figsize=(8, 6))

# Apply a larger offset for separation
plt.plot(time_vector, mean_eye_trace_low_avg + 5, color='blue', label='Low Pupil Diameter (Offset)')
plt.fill_between(time_vector, 
                 mean_eye_trace_low_avg + 5 - stderr_eye_trace_low_avg, 
                 mean_eye_trace_low_avg + 5 + stderr_eye_trace_low_avg, 
                 color='blue', alpha=0.2)

plt.plot(time_vector, mean_eye_trace_high_avg - 5, color='red', label='High Pupil Diameter (Offset)')
plt.fill_between(time_vector, 
                 mean_eye_trace_high_avg - 5 - stderr_eye_trace_high_avg, 
                 mean_eye_trace_high_avg - 5 + stderr_eye_trace_high_avg, 
                 color='red', alpha=0.2)

plt.axvline(x=0, color='black', linestyle='--')  # Mark stimulus onset
plt.xlabel('Time (s)')
plt.ylabel('Pupil Diameter')
plt.legend()
plt.show()

# Save the average results for all datasets (optional)
# np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_low_all_datasets.npy'), mean_eye_trace_low_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_high_all_datasets.npy'), mean_eye_trace_high_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'stderr_eye_trace_low_all_datasets.npy'), stderr_eye_trace_low_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'stderr_eye_trace_high_all_datasets.npy'), stderr_eye_trace_high_avg)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import zscore
from utils import utils as ut
from lib import readSGLX
import os

# Constants
pre_time = 0.15  # Pre-stimulus window (-0.15s)
post_time = 0.15  # Post-stimulus window (+0.15s)
chanList = [2]   
num_datasets = 6  # Number of datasets to process

# Function to calculate z-score
def z_score_normalization(data):
    return zscore(data, axis=None)

# Load trial start/stop times from trial DataFrame
trial_DF_path = ut.getFilePath(windowTitle="Trial CSV", filetypes=[("CSV file", "*.csv")])
trialDF = pd.read_csv(trial_DF_path)

# Load stimulus start/stop times from stimulus DataFrame
stimulusDF_path = ut.getFilePath(windowTitle="Stimulus CSV", filetypes=[("CSV file", "*.csv")])
stimulusDF = pd.read_csv(stimulusDF_path)

# Use the correct column names 'trialstart' and 'trialstop' from trialDF
trial_starts = trialDF['trialstart'].values
trial_stops = trialDF['trialstop'].values

# Use 'stimstart' and 'stimstop' from stimulusDF
stim_starts = stimulusDF['stimstart'].values
stim_stops = stimulusDF['stimstop'].values

# Median pupil diameter calculation using trial and stimulus information
def calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sample_rate):
    medians = []
    for trial_start, trial_stop in zip(trial_starts, trial_stops):
        trial_medians = []
        for stim_start, stim_stop in zip(stim_starts, stim_stops):
            if stim_start >= trial_start and stim_stop <= trial_stop:
                start_sample = int((stim_start - pre_time) * sample_rate)
                stop_sample = int((stim_stop + post_time) * sample_rate)
                trial_data = pupil_diameter[start_sample:stop_sample]
                median_diameter = np.median(trial_data)
                trial_medians.append(median_diameter)
        if trial_medians:
            medians.append(np.median(trial_medians))
    return np.array(medians)

# Initialize lists to hold averages for all datasets
all_mean_eye_trace_low = []
all_mean_eye_trace_high = []
all_stderr_eye_trace_low = []
all_stderr_eye_trace_high = []

# Loop over 6 datasets
for i in range(num_datasets):
    print(f"Processing dataset {i+1}/{num_datasets}...")
    
    # Load binary file and metadata
    binFullPath = ut.getFilePath(windowTitle=f"Binary nidq file {i+1}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)

    # Load pupil diameter data
    pupilFullPath = ut.getFilePath(windowTitle=f"Pupil diameter data {i+1}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)

    # Calculate median pupil diameter for each trial
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sRate)

    # Split into low and high pupil diameter trials
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Trial-based pupil trace extraction (-0.15 to +0.15s around stimulus onset)
    trial_eye_traces = []
    time_window_samples = int((pre_time + post_time) * sRate)  # Total samples in window

    for stim_start in stim_starts:
        stim_index = int(stim_start * sRate)
        pre_stim_index = stim_index - int(pre_time * sRate)
        post_stim_index = stim_index + int(post_time * sRate)
        
        trial_eye_trace = np.full(time_window_samples, np.nan)
        valid_start = max(0, pre_stim_index)
        valid_end = min(len(pupil_diameter), post_stim_index)
        insert_start = max(0, -pre_stim_index)
        insert_end = insert_start + (valid_end - valid_start)
        
        trial_eye_trace[insert_start:insert_end] = pupil_diameter[valid_start:valid_end]
        trial_eye_traces.append(trial_eye_trace)

    # Convert to NumPy array
    trial_eye_traces = np.array(trial_eye_traces)

    # Separate into low and high pupil diameter trials
    low_pupil_traces = trial_eye_traces[inds_low]
    high_pupil_traces = trial_eye_traces[inds_high]

    # Z-score normalization
    low_pupil_traces = z_score_normalization(low_pupil_traces)
    high_pupil_traces = z_score_normalization(high_pupil_traces)

    # Compute average pupil diameter across low and high trials
    mean_eye_trace_low = np.nanmean(low_pupil_traces, axis=0)
    mean_eye_trace_high = np.nanmean(high_pupil_traces, axis=0)

    # Compute standard error for low and high trials
    stderr_eye_trace_low = np.nanstd(low_pupil_traces, axis=0) / np.sqrt(len(low_pupil_traces))
    stderr_eye_trace_high = np.nanstd(high_pupil_traces, axis=0) / np.sqrt(len(high_pupil_traces))

    # Store results for final average calculation across datasets
    all_mean_eye_trace_low.append(mean_eye_trace_low)
    all_mean_eye_trace_high.append(mean_eye_trace_high)
    all_stderr_eye_trace_low.append(stderr_eye_trace_low)
    all_stderr_eye_trace_high.append(stderr_eye_trace_high)

# Compute average across datasets
mean_eye_trace_low_avg = np.mean(all_mean_eye_trace_low, axis=0)
mean_eye_trace_high_avg = np.mean(all_mean_eye_trace_high, axis=0)
stderr_eye_trace_low_avg = np.mean(all_stderr_eye_trace_low, axis=0)
stderr_eye_trace_high_avg = np.mean(all_stderr_eye_trace_high, axis=0)

# Time vector from -0.15 to +0.15s
time_vector = np.linspace(-pre_time, post_time, len(mean_eye_trace_low_avg))

# Plot the average z-score pupil diameter trace for low and high conditions across all datasets
plt.figure(figsize=(8, 6))

plt.plot(time_vector, mean_eye_trace_low_avg + 5, color='blue', label='Low Pupil Diameter (Offset, Z-score)')
plt.fill_between(time_vector, 
                 mean_eye_trace_low_avg + 5 - stderr_eye_trace_low_avg, 
                 mean_eye_trace_low_avg + 5 + stderr_eye_trace_low_avg, 
                 color='blue', alpha=0.2)

plt.plot(time_vector, mean_eye_trace_high_avg - 5, color='red', label='High Pupil Diameter (Offset, Z-score)')
plt.fill_between(time_vector, 
                 mean_eye_trace_high_avg - 5 - stderr_eye_trace_high_avg, 
                 mean_eye_trace_high_avg - 5 + stderr_eye_trace_high_avg, 
                 color='red', alpha=0.2)

plt.axvline(x=0, color='black', linestyle='--')  # Mark stimulus onset
plt.xlabel('Time (s)')
plt.ylabel('Z-score Pupil Diameter')
plt.legend()
plt.show()

# Save the results for each dataset (optional)
# np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_low_zscore.npy'), mean_eye_trace_low_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_high_zscore.npy'), mean_eye_trace_high_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'stderr_eye_trace_low_zscore.npy'), stderr_eye_trace_low_avg)
# np.save(os.path.join(os.path.dirname(binFullPath), 'stderr_eye_trace_high_zscore.npy'), stderr_eye_trace_high_avg)

from lib import readSGLX
import numpy as np
import matplotlib.pyplot as plt
from utils import utils as ut
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq
import os

# Constants
tStart = 0        
tEnd = 60.0
chanList = [2]   
thr = 12
record_start = 0
record_stop  = -1

# Function to detect blinks based on the pupil diameter
def detect_blinks(pupil_diameters, threshold_factor=5):
    derivative = np.diff(pupil_diameters)
    std_dev = np.std(derivative)
    blink_indices = np.where(np.abs(derivative) > threshold_factor * std_dev)[0]
    return blink_indices

# Replace blinks with NaN values
def replace_blinks_with_nan(pupil_diameters, blink_indices, sample_rate, mask_length=0.1):
    range_around_blink = int(mask_length * sample_rate)
    for blink_index in blink_indices:
        start = max(0, blink_index - range_around_blink)
        end = min(len(pupil_diameters), blink_index + range_around_blink)
        pupil_diameters[start:end] = np.nan
    return pupil_diameters

# Interpolate missing values
def interpolate_nans(pupil_diameters):
    nan_indices = np.where(np.isnan(pupil_diameters))[0]
    non_nan_indices = np.where(~np.isnan(pupil_diameters))[0]
    pupil_diameters[nan_indices] = np.interp(nan_indices, non_nan_indices, pupil_diameters[non_nan_indices])
    return pupil_diameters

# FFT plot for frequency analysis
def plot_fft_limited(data, sample_rate, title, max_freq=100):
    data = detrend(data)  # Remove any linear trend
    n = len(data)
    freqs = fftfreq(n, d=1/sample_rate)
    fft_data = fft(data)
    magnitude = np.abs(fft_data)
    
    # Select only the frequency range up to max_freq
    mask = freqs[:n // 2] <= max_freq
    
    plt.figure()
    plt.plot(freqs[:n // 2][mask], magnitude[:n // 2][mask])
    plt.title(f'Fourier Transform of {title}')
    plt.xlim(0, max_freq)  # Ensure x-axis is limited to 0-50 Hz
    plt.show()

# Low-pass filter the data
def low_pass_filter(data, sample_rate, cutoff=5.0, order=4):
    data = detrend(data)  # Remove trend
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Normalize the data
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function to calculate the standard error of the data
def calculate_standard_error(data):
    std_dev = np.nanstd(data, axis=0)  # Handle NaN values
    n = np.sum(~np.isnan(data), axis=0)  # Count non-NaN values for each point
    return std_dev / np.sqrt(n)

# Load the data
binFullPath = ut.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

firstSamp = int(sRate * tStart)
lastSamp = int(sRate * tEnd)
rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
selectData = rawData[chanList, firstSamp:lastSamp+1]
del rawData

pupilDiameter = 1e3 * readSGLX.GainCorrectNI(selectData, chanList, meta)
pupilDiameter = pupilDiameter[0, record_start:record_stop]

# Load trial start and stop times from the trial DataFrame
trial_DF_path = ut.getFilePath(windowTitle="Trial CSV", filetypes=[("CSV file", "*.csv")])
trialDF = pd.read_csv(trial_DF_path)

trial_starts = trialDF['trialstart'].values  # Assuming trialstart column exists
trial_stops = trialDF['trialstop'].values    # Assuming trialstop column exists

# Plot raw data before any processing
plt.figure(figsize=(10, 4))
plt.plot(pupilDiameter, label="Raw Pupil Diameter")
plt.title('Raw Pupil Diameter Data')

# Plot trial start and stop as vertical lines
for start, stop in zip(trial_starts, trial_stops):
    plt.axvline(x=start * sRate, color='green', linestyle='--', label='Trial Start' if start == trial_starts[0] else "")
    plt.axvline(x=stop * sRate, color='red', linestyle='--', label='Trial Stop' if stop == trial_stops[0] else "")

plt.xlabel('Time (samples)')
plt.legend()
plt.show()

# Detect and remove blinks
blink_inds = detect_blinks(pupilDiameter, thr)
pupilDiameter = replace_blinks_with_nan(pupilDiameter, blink_inds, sRate)
pupilDiameter = interpolate_nans(pupilDiameter)

# Start the data at 90k samples but keep the y-axis scale
pupilDiameter = pupilDiameter[90000:]

# Plot raw data starting from 90k samples
plt.figure(figsize=(10, 4))
plt.plot(pupilDiameter, label="Pupil Diameter After Blink Removal")
plt.title('Pupil Diameter Data Starting from 90k Samples')

# Plot trial start and stop again on the processed data
for start, stop in zip(trial_starts, trial_stops):
    plt.axvline(x=start * sRate - 90000, color='green', linestyle='--', label='Trial Start' if start == trial_starts[0] else "")
    plt.axvline(x=stop * sRate - 90000, color='red', linestyle='--', label='Trial Stop' if stop == trial_stops[0] else "")

plt.xlabel('Time (samples)')
plt.legend()
plt.show()

# Plot the FFT of the raw data after blink removal
plot_fft_limited(pupilDiameter, sRate, 'Pupil Diameter After Blink Removal', max_freq=50)

# Apply low-pass filter with detrending
cutoff_frequency = 5.0  
pupilDiameter_filtered = low_pass_filter(pupilDiameter, sRate, cutoff=cutoff_frequency)

# Plot filtered data with trial start and stop times
plt.plot(pupilDiameter_filtered, label="Filtered Pupil Diameter")
plt.title('Low-Pass Filtered Pupil Diameter')

for start, stop in zip(trial_starts, trial_stops):
    plt.axvline(x=start * sRate - 90000, color='green', linestyle='--', label='Trial Start' if start == trial_starts[0] else "")
    plt.axvline(x=stop * sRate - 90000, color='red', linestyle='--', label='Trial Stop' if stop == trial_stops[0] else "")

plt.xlabel('Time (samples)')
plt.ylim(-400, 1200)  # Keep y-axis limits
plt.legend()
plt.show()

# Normalize the detrended and filtered data
pupilDiameter_normalized = normalize_data(pupilDiameter_filtered)

# Calculate standard error for the pupil diameter data
std_err = calculate_standard_error(pupilDiameter_normalized)

# Plot limited normalized data with standard error
time_limit = 60 * sRate  # 60 seconds worth of samples
plt.figure(figsize=(10, 6))
plt.plot(pupilDiameter_normalized[:int(time_limit)], label="Normalized Pupil Diameter")

# Plot standard error band (mean ± standard error)
plt.fill_between(np.arange(len(pupilDiameter_normalized[:int(time_limit)])), 
                 pupilDiameter_normalized[:int(time_limit)] - std_err, 
                 pupilDiameter_normalized[:int(time_limit)] + std_err, 
                 color='blue', alpha=0.2, label="Standard Error")

plt.xlabel('Time (samples)')
plt.ylabel('Pupil Diameter')

# Plot trial start/stop only for the selected time window
for start, stop in zip(trial_starts, trial_stops):
    if start * sRate < time_limit:
        plt.axvline(x=start * sRate, color='green', linestyle='--', label='Trial Start' if start == trial_starts[0] else "")
        plt.axvline(x=stop * sRate, color='red', linestyle='--', label='Trial Stop' if stop == trial_stops[0] else "")

plt.legend()
plt.show()

# Plot the FFT of the filtered data
plot_fft_limited(pupilDiameter_filtered, sRate, 'Filtered Pupil Diameter', max_freq=50)

# Save the processed data
flname = os.path.join(os.path.dirname(binFullPath), 'processed_pupil_diameter_filtered.npy')
np.save(flname, pupilDiameter_filtered)


# this script needs to be run from the terminal, not interactively
from lib import readSGLX
import numpy as np
import matplotlib.pyplot as plt
from utils import utils as ut # type: ignore
from scipy.signal import detrend
import os

tStart = 0        
tEnd = 677.078
chanList = [2]   
thr = 3
record_start = 0
record_stop  = -1

def detect_blinks(pupil_diameters, threshold_factor=5):
    # Calculate the derivative of the pupil diameters
    derivative = np.diff(pupil_diameters)

    # Calculate the standard deviation of the derivative
    std_dev = np.std(derivative)

    # Identify the indices where the absolute value of the derivative exceeds the threshold
    blink_indices = np.where(np.abs(derivative) > threshold_factor * std_dev)[0]

    return blink_indices

def replace_blinks_with_nan(pupil_diameters, blink_indices,sample_rate,mask_length=0.1):
    # Define the range around each blink to replace with NaNs
    range_around_blink = int(mask_length*sample_rate)

    # For each blink index, replace the range around it with NaNs
    for blink_index in blink_indices:
        start = max(0, blink_index - range_around_blink)
        end = min(len(pupil_diameters), blink_index + range_around_blink)
        pupil_diameters[start:end] = np.nan

    return pupil_diameters

def interpolate_nans(pupil_diameters):
    # Identify the indices of the NaN values
    nan_indices = np.where(np.isnan(pupil_diameters))[0]

    # Identify the indices of the non-NaN values
    non_nan_indices = np.where(~np.isnan(pupil_diameters))[0]

    # Interpolate the NaN values
    pupil_diameters[nan_indices] = np.interp(nan_indices, non_nan_indices, pupil_diameters[non_nan_indices])

    return pupil_diameters

def interactive_plot(pupil_diameters):
    from matplotlib.widgets import SpanSelector, Button
    
    # Function to be called when an interval is selected
    def onselect(xmin, xmax):
        # Convert the x-values to indices
        imin, imax = int(xmin), int(xmax)
        # Replace the selected interval with NaNs
        pupil_diameters[imin:imax] = np.nan

    # Function to be called when the button is pressed
    def on_button_press(event):
        plt.close(fig)
    
    fig, ax = plt.subplots()

    # Plot the pupil diameters
    ax.plot(pupil_diameters)

    # Create a SpanSelector
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                        props=dict(alpha=0.5, facecolor='red'))

    # Create a Button
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Done', color='lightgoldenrodyellow', hovercolor='0.975')
    button.on_clicked(on_button_press)

    plt.show()

    return pupil_diameters

# Load the data
binFullPath = ut.getFilePath(windowTitle="Binary nidq file",filetypes=[("NIdq binary","*.bin")]) # type: ignore
meta  = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

firstSamp = int(sRate*tStart)
lastSamp = int(sRate*tEnd)
rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
selectData = rawData[chanList, firstSamp:lastSamp+1]
del rawData
pupilDiameter = 1e3*readSGLX.GainCorrectNI(selectData, chanList, meta)

pupilDiameter = pupilDiameter[0,record_start:record_stop]

blink_inds = detect_blinks(pupilDiameter, thr)
pupilDiameter = replace_blinks_with_nan(pupilDiameter, blink_inds,sRate)
pupilDiameter = interpolate_nans(pupilDiameter)
pupilDiameter = interactive_plot(pupilDiameter)
#pupilDiameter[0:80000] = np.nan
pupilDiameter = interpolate_nans(pupilDiameter)

plt.plot(pupilDiameter)
plt.show()
flname = os.path.join(binFullPath.parent,'processed_pupil_diameter.npy')
np.save(flname,pupilDiameter)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from utils import utils as ut
from lib import readSGLX
import os

# Constants
pre_time = 0.15  # Pre-stimulus window (-0.15s)
post_time = 0.15  # Post-stimulus window (+0.15s)
chanList = [2]   

# Load the data
binFullPath = ut.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary","*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

# Load pupil diameter data
pupilFullPath = ut.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", '*.npy')])
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)

# Load trial start/stop times from trial DataFrame
trial_DF_path = ut.getFilePath(windowTitle="Trial CSV", filetypes=[("CSV file", "*.csv")])
trialDF = pd.read_csv(trial_DF_path)

# Load stimulus start/stop times from stimulus DataFrame
stimulusDF_path = ut.getFilePath(windowTitle="Stimulus CSV", filetypes=[("CSV file", "*.csv")])
stimulusDF = pd.read_csv(stimulusDF_path)

# Use the correct column names 'trialstart' and 'trialstop' from trialDF
trial_starts = trialDF['trialstart'].values
trial_stops = trialDF['trialstop'].values

# Use 'stimstart' and 'stimstop' from stimulusDF
stim_starts = stimulusDF['stimstart'].values
stim_stops = stimulusDF['stimstop'].values

# Median pupil diameter calculation using trial and stimulus information
def calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sample_rate):
    medians = []
    for trial_start, trial_stop in zip(trial_starts, trial_stops):
        # For each trial, find the corresponding stimulus windows within that trial
        trial_medians = []
        for stim_start, stim_stop in zip(stim_starts, stim_stops):
            if stim_start >= trial_start and stim_stop <= trial_stop:
                start_sample = int((stim_start - pre_time) * sample_rate)
                stop_sample = int((stim_stop + post_time) * sample_rate)
                trial_data = pupil_diameter[start_sample:stop_sample]
                median_diameter = np.median(trial_data)
                trial_medians.append(median_diameter)
        if trial_medians:
            medians.append(np.median(trial_medians))
    return np.array(medians)

# Calculate median pupil diameter for each trial based on trial and stimulus windows
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, trial_starts, trial_stops, stim_starts, stim_stops, sRate)

# Split into low and high pupil diameter trials
pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Trial-based pupil trace extraction (-0.15 to +0.15s around stimulus onset)
trial_eye_traces = []
time_window_samples = int((pre_time + post_time) * sRate)  # Total samples in window

for stim_start in stim_starts:
    stim_index = int(stim_start * sRate)  # Convert stimulus start time to sample index
    pre_stim_index = stim_index - int(pre_time * sRate)  # Start window at -0.15s
    post_stim_index = stim_index + int(post_time * sRate)  # End window at +0.15s
    
    # Initialize array with NaNs for padding out-of-bound indices
    trial_eye_trace = np.full(time_window_samples, np.nan)
    
    # Valid indices to insert the pupil diameter data
    valid_start = max(0, pre_stim_index)
    valid_end = min(len(pupil_diameter), post_stim_index)
    
    # Insert the valid pupil diameter data into the trial trace
    insert_start = max(0, -pre_stim_index)
    insert_end = insert_start + (valid_end - valid_start)
    
    trial_eye_trace[insert_start:insert_end] = pupil_diameter[valid_start:valid_end]
    trial_eye_traces.append(trial_eye_trace)

# Convert to NumPy array for easy processing
trial_eye_traces = np.array(trial_eye_traces)

# Separate into low and high pupil diameter trials
low_pupil_traces = trial_eye_traces[inds_low]
high_pupil_traces = trial_eye_traces[inds_high]

# Compute average pupil diameter across low and high trials
mean_eye_trace_low = np.nanmean(low_pupil_traces, axis=0)
mean_eye_trace_high = np.nanmean(high_pupil_traces, axis=0)
time_vector = np.linspace(-pre_time, post_time, len(mean_eye_trace_low))  # Time vector from -0.15 to +0.15s

# Plot the average pupil diameter trace for low and high conditions
plt.figure(figsize=(8, 6))
plt.plot(time_vector, mean_eye_trace_low, color='blue', label='Low Pupil Diameter')
plt.plot(time_vector, mean_eye_trace_high, color='red', label='High Pupil Diameter')
plt.axvline(x=0, color='black', linestyle='--')  # Mark stimulus onset
plt.xlabel('Time (s)')
plt.ylabel('Pupil Diameter')
plt.legend()
plt.show()

# Save the results
np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_low.npy'), mean_eye_trace_low)
np.save(os.path.join(os.path.dirname(binFullPath), 'average_eye_trace_high.npy'), mean_eye_trace_high)


# This script needs to be run with the pypill environment activated
from lib import readSGLX
import numpy as np
import matplotlib.pyplot as plt
from utils import utils as ut # type: ignore
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq
import os

tStart = 0        
tEnd = 60.0
chanList = [2]   
thr = 12
record_start = 0
record_stop  = -1

def detect_blinks(pupil_diameters, threshold_factor=5):
    derivative = np.diff(pupil_diameters)
    std_dev = np.std(derivative)
    blink_indices = np.where(np.abs(derivative) > threshold_factor * std_dev)[0]
    return blink_indices

def replace_blinks_with_nan(pupil_diameters, blink_indices, sample_rate, mask_length=0.1):
    range_around_blink = int(mask_length * sample_rate)
    for blink_index in blink_indices:
        start = max(0, blink_index - range_around_blink)
        end = min(len(pupil_diameters), blink_index + range_around_blink)
        pupil_diameters[start:end] = np.nan
    return pupil_diameters

def interpolate_nans(pupil_diameters):
    nan_indices = np.where(np.isnan(pupil_diameters))[0]
    non_nan_indices = np.where(~np.isnan(pupil_diameters))[0]
    pupil_diameters[nan_indices] = np.interp(nan_indices, non_nan_indices, pupil_diameters[non_nan_indices])
    return pupil_diameters

def plot_fft_limited(data, sample_rate, title, max_freq=100):
    data = detrend(data)  # Remove any linear trend
    n = len(data)
    freqs = fftfreq(n, d=1/sample_rate)
    fft_data = fft(data)
    magnitude = np.abs(fft_data)
    
    # Select only the frequency range up to max_freq
    mask = freqs[:n // 2] <= max_freq
    
    plt.figure()
    plt.plot(freqs[:n // 2][mask], magnitude[:n // 2][mask])
    plt.title(f'Fourier Transform of {title}')
    plt.xlim(0, max_freq)  # Ensure x-axis is limited to 0-50 Hz
    plt.show()

def low_pass_filter(data, sample_rate, cutoff=5.0, order=4):
    # Detrend the data before filtering
    data = detrend(data)  
    
    # Calculate the Nyquist frequency
    nyquist = 0.5 * sample_rate
    
    # Normalize the cutoff frequency
    normal_cutoff = cutoff / nyquist
    
    # Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Load the data
binFullPath = ut.getFilePath(windowTitle="Binary nidq file",filetypes=[("NIdq binary","*.bin")]) # type: ignore
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

firstSamp = int(sRate * tStart)
lastSamp = int(sRate * tEnd)
rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
selectData = rawData[chanList, firstSamp:lastSamp+1]
del rawData
pupilDiameter = 1e3 * readSGLX.GainCorrectNI(selectData, chanList, meta)

pupilDiameter = pupilDiameter[0, record_start:record_stop]

# Plot the raw data before any processing
plt.figure(figsize=(10, 4))
plt.plot(pupilDiameter)
plt.title('Raw Pupil Diameter Data')
plt.xlabel('Time (samples)')
plt.show()

# Detect and remove blinks
blink_inds = detect_blinks(pupilDiameter, thr)
pupilDiameter = replace_blinks_with_nan(pupilDiameter, blink_inds, sRate)
pupilDiameter = interpolate_nans(pupilDiameter)

# Start the data at 90k samples but keep the y-axis scale
pupilDiameter = pupilDiameter[90000:]

# Plot the raw data starting from 90k samples with unchanged y-axis
plt.figure(figsize=(10, 4))
plt.plot(pupilDiameter)
plt.title('Raw Pupil Diameter Data Starting from 90k Samples')
plt.xlabel('Time (samples)')  # Keep the y-axis limits as in the filtered plot
plt.show()

# Plot the FFT of the raw data after blink removal
plot_fft_limited(pupilDiameter, sRate, 'Raw Pupil Diameter After Blink Removal', max_freq=50)

# Apply low-pass filter with detrending
cutoff_frequency = 5.0  
pupilDiameter_filtered = low_pass_filter(pupilDiameter, sRate, cutoff=cutoff_frequency)

# Plot the filtered data starting from 90k samples with unchanged y-axis
plt.plot(pupilDiameter_filtered)
plt.title(' Low-Pass Filtering')
plt.xlabel('Time (samples)')
plt.ylim(-400, 1200)  # Keep the y-axis limits as in the provided plot
plt.show()

# Normalize the detrended and filtered data
pupilDiameter_normalized = normalize_data(pupilDiameter_filtered)

# Plot the detrended and normalized data
plt.plot(pupilDiameter_normalized)
plt.xlabel('Time (samples)')
plt.ylabel('Pupil Diameter')
plt.show()

# Plot the FFT of the filtered data
plot_fft_limited(pupilDiameter_filtered, sRate, 'Filtered Pupil Diameter', max_freq=50)

# Save the processed data
flname = os.path.join(os.path.dirname(binFullPath), 'processed_pupil_diameter_filtered.npy')
np.save(flname, pupilDiameter_filtered)

