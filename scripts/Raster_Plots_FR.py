from utils import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_DF_path = utils.getFilePath(windowTitle="trials_DF_Fixation", filetypes=[('Trials_Fixation csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
trialDF = pd.read_csv(trial_DF_path)

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def plot_raster(Y, start_times, stop_times, firing_rates, cluster_index):
    fig, ax = plt.subplots()
    for trial, (start_time, stop_time, firing_rate) in enumerate(zip(start_times, stop_times, firing_rates)):
        spikes_in_trial = Y[(Y >= start_time - 0.2) & (Y <= stop_time)]
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=trial+1)
    ax.set_title(f'Raster Plot - Cluster {cluster_index}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    plt.savefig(f'Cluster_{cluster_index}_raster.svg')  # Save the raster plot for each cluster
    plt.close()  # Close the plot to avoid overlapping

def calculate_mean_firing_rate(Y, trialDF, i):
    start_times = trialDF['fixationstart']
    stop_times = trialDF['fixationstop']
    firing_rates = []

    for start_time, stop_time in zip(start_times, stop_times):
        spikes_in_window = Y[(Y >= start_time - 0.2) & (Y <= stop_time)]
        firing_rate = len(spikes_in_window) / (stop_time - start_time) if stop_time > start_time else 0
        firing_rates.append(firing_rate)

    plot_raster(Y, start_times, stop_times, firing_rates, i)

for i, Y in spike_times_clusters.items():
    calculate_mean_firing_rate(Y, trialDF, i)
