# Calculate mean spike count and add to DataFrame
spike_times_sec = np.load(spikesFullPath.with_stem('spike_times_sec.npy'))
trialsDF['spike_count'] = 0  # Initialize a new column for spike count
trialsDF['mean_firing_rate'] = 0  # Initialize a new column for mean firing rate

# Iterate through each trial and calculate spike count and mean firing rate
for index, row in trialsDF.iterrows():
    start_time = row['fixationstart']
    stop_time = row['fixationstop']

    # Filter spike times within the trial window
    spike_count = np.sum((spike_times_sec >= start_time) & (spike_times_sec <= stop_time))

    # Calculate trial duration in seconds
    trial_duration = stop_time - start_time

    # Calculate mean firing rate (spike count/trial duration)
    mean_firing_rate = spike_count / trial_duration if trial_duration > 0 else 0

    # Update the DataFrame
    trialsDF.at[index, 'spike_count'] = spike_count
    trialsDF.at[index, 'mean_firing_rate'] = mean_firing_rate

# Save the updated DataFrame
trialsDF.to_csv(trialstartFullPath.with_suffix('.csv'), index=False)
