
# This script needs to be run with the pypill environment activated
from lib import readSGLX
import numpy as np
import matplotlib.pyplot as plt
from utils import utils as ut # type: ignore
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq
import os
s
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

