import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib import readSGLX
import os

# Define the results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data paths using utility functions
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])
pupil_diameter_voltage_path = utils.getFilePath(windowTitle="Pupil Diameter Data", filetypes=[('Numpy file', '*.npy')])

# Load actual data
spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter_voltage = np.load(pupil_diameter_voltage_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

min_voltage = pupil_diameter_voltage.min()
max_voltage = pupil_diameter_voltage.max()

print(f"Voltage Data Range: {min_voltage} to {max_voltage}")


normalized_voltage = (pupil_diameter_voltage - min_voltage) / (max_voltage - min_voltage)

pixel_horizontal = normalized_voltage * 255
pixel_vertical = normalized_voltage * 128


pupil_diameter_pixels = (pixel_horizontal + pixel_vertical) / 2

def calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter_pixels[start_sample:stop_sample]
        medians.append(np.median(trial_data))
    return np.array(medians)


trial_medians = calculate_median_pupil_diameter(pupil_diameter_pixels, stimulusDF, baseline_time=0.15, sample_rate=sRate)

overall_median = np.median(trial_medians)
low_trials = np.where(trial_medians < overall_median)[0]
high_trials = np.where(trial_medians >= overall_median)[0]

def calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, trial_indices, sample_rate):
    avg_pupil_diameters = []
    for trial in trial_indices:
        start_sample = int(stimulusDF.iloc[trial]['stimstart'] * sample_rate)
        stop_sample = int(stimulusDF.iloc[trial]['stimstop'] * sample_rate)
        avg_pupil_diameters.append(np.mean(pupil_diameter_pixels[start_sample:stop_sample]))
    return np.mean(avg_pupil_diameters)


avg_low = calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, low_trials, sRate)
avg_high = calculate_average_pupil_diameter(pupil_diameter_pixels, stimulusDF, high_trials, sRate)

print(f"Average Pupil Diameter for Low Trials: {avg_low:.2f} pixels")
print(f"Average Pupil Diameter for High Trials: {avg_high:.2f} pixels")




