import matplotlib.pyplot as plt

# Filter out only the relevant columns for visualization
classification_results = results_df[['Cluster', 'Baseline Classification', 'Evoked Classification']]

# Count the occurrences of each classification type for Baseline and Evoked
baseline_counts = classification_results['Baseline Classification'].value_counts()
evoked_counts = classification_results['Evoked Classification'].value_counts()

# Create subplots to visualize the counts
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for Baseline Classification
axs[0].bar(baseline_counts.index, baseline_counts.values, color='skyblue')
axs[0].set_title('Baseline Classification Counts')
axs[0].set_xlabel('Classification')
axs[0].set_ylabel('Number of Clusters')

# Plot for Evoked Classification
axs[1].bar(evoked_counts.index, evoked_counts.values, color='lightgreen')
axs[1].set_title('Evoked Classification Counts')
axs[1].set_xlabel('Classification')
axs[1].set_ylabel('Number of Clusters')

plt.tight_layout()
plt.show()











import seaborn as sns

# Create a dataframe for firing rate analysis
firing_rate_data = results_df[['Baseline High Firing Rate', 'Baseline Low Firing Rate',
                               'Evoked High Firing Rate', 'Evoked Low Firing Rate', 
                               'Baseline Classification', 'Evoked Classification']]

# Melt the dataframe to get it in a long format suitable for seaborn
firing_rate_melted = firing_rate_data.melt(id_vars=['Baseline Classification', 'Evoked Classification'],
                                           value_vars=['Baseline High Firing Rate', 'Baseline Low Firing Rate',
                                                       'Evoked High Firing Rate', 'Evoked Low Firing Rate'],
                                           var_name='Firing Rate Type', value_name='Firing Rate')

# Plotting
plt.figure(figsize=(14, 8))
sns.boxplot(x='Firing Rate Type', y='Firing Rate', hue='Baseline Classification', data=firing_rate_melted)
plt.title('Distribution of Firing Rates by Baseline Classification')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(14, 8))
sns.boxplot(x='Firing Rate Type', y='Firing Rate', hue='Evoked Classification', data=firing_rate_melted)
plt.title('Distribution of Firing Rates by Evoked Classification')
plt.xticks(rotation=45)
plt.show()
