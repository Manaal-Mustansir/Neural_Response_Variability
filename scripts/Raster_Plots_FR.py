from utils import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load spike times, clusters, and trial DataFrame
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_DF_path = utils.getFilePath(windowTitle="trials_DF_Fixation", filetypes=[('Trials_Fixation csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
trialDF = pd.read_csv(trial_DF_path)

# Extract spike times for each cluster
spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

# Extract data for the first three trials
trialDF_first_three = trialDF.head(3)

fix, ax = plt.subplots(figsize=(10, len(spike_times_clusters) * 0.025))

for i, (cluster_index, Y) in enumerate(spike_times_clusters.items()):
    for index, row in trialDF_first_three.iterrows():
        start_time = row['fixationstart']
        stop_time = row['fixationstop']

        # Filter spike times within the trial window
        spikes_in_trial = Y[(Y >= start_time - 0.2) & (Y <= stop_time)]

        # Plot raster plot
        ax.eventplot(spikes_in_trial - start_time, color='black', linewidths=0.5, lineoffsets=i+1)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Clusters')
ax.set_title('All Clusters')
plt.show()
