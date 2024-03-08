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


def plot_raster(Y, start_time, stop_time, firing_rate, cluster_index, ax, trial):
    # Plot raster plot
    ax.eventplot(Y - start_time, color='black', linewidths=0.5, lineoffsets=trial+1)
    ax.set_title(f'Raster Plot - Cluster {cluster_index}')
    ax.set_label('Time (s)')
    ax.set_label('Firing Rate (spikes/s)')

    # Plot firing rate on a separate axis
    """ ax1 = plt.gca().twinx()
    ax1.plot([start_time, stop_time], [firing_rate, firing_rate], color='red', linestyle='dashed', linewidth=2)
    ax1.set_ylabel('Firing Rate (spikes/s)', color='red')
    ax1.tick_params(axis='y', labelcolor='red') """


def calculate_mean_firing_rate(Y, trialDF, i):
    firing_rates = []  # List to store firing rates for each trial
    fix,ax = plt.subplots(1,1)
    for index, row in trialDF.iterrows():
        start_time = row['fixationstart']
        stop_time = row['fixationstop']

        # Filter spike times within the trial window
        spikes_in_trial = Y[(Y >= start_time- 0.2) & (Y <= stop_time)]

        # Calculate firing rate for the trial
        firing_rate = len(spikes_in_trial) / (stop_time - start_time) if stop_time > start_time else 0

        # Update the firing rates list
        firing_rates.append(firing_rate)

        # Calculate baseline window
        baseline_start = start_time - 0.2  # 200ms before fixation start
        baseline_end = start_time

        # Update the DataFrame
        trialDF.at[index, f'baseline_spike_count_cluster_{i}'] = len(spikes_in_trial)
        trialDF.at[index, f'baseline_firing_rate_cluster_{i}'] = len(spikes_in_trial) / (baseline_end - baseline_start) if baseline_end > baseline_start else 0
        trialDF.at[index, f'evoked_response_cluster_{i}'] = firing_rate

            # Pass firing_rate as an argument to the plot_raster function
        plot_raster(spikes_in_trial, start_time, stop_time, firing_rate, i, ax, trial=index)
    plt.show()

    return trialDF

for i in spike_times_clusters.keys():
    Y = spike_times_clusters[i]

    # Calculate mean firing rate, baseline, and evoked response for each cluster
    trialDF = calculate_mean_firing_rate(Y, trialDF, i)

# Display the updated DataFrame
print(trialDF)