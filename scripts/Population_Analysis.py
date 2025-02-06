#ORIGINAL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Constants for penetration date and monkey name
penetration_date = '2024-05-29'
monkey_name = 'Sansa'

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

# Store results from all datasets
all_results = []

# Loop through six datasets
for dataset_num in range(1, 7):  # Loop through the desired dataset(s)
    print(f"Processing dataset {dataset_num}...")

    # Load data for each dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])

    # Load spike times and cluster data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)

    # Load the trial mat file and extract stimuli
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']

    def linear_stimuli(expt_info):
        stims = np.array([])
        for i in range(expt_info.trial_records.shape[0]):
            stims = np.hstack((stims, expt_info.trial_records[i].trImage))
        return stims

    stimuli = linear_stimuli(expt_info)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]  # Removing NaN values from stimuli

    # Only keep "stimulus 1"
    stimulusDF = stimulusDF[stimulusDF['stimuli'] == 1]

    trialDF = pd.read_csv(trial_DF_path)
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    # Calculate median pupil diameter
    def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
        medians = []
        for index, row in stimulusDF.iterrows():
            start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
            stop_sample = int(row['stimstop'] * sample_rate)
            trial_data = pupil_diameter[start_sample:stop_sample]
            median_diameter = np.median(trial_data)
            medians.append(median_diameter)
        return np.array(medians)

    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)

    # Define range around the median to remove
    threshold = 0.04 * pup_median  # Adjust percentage 
    lower_bound = pup_median - threshold
    upper_bound = pup_median + threshold

    # Exclude trials near the median
    inds_low = np.where((pupil_trials < lower_bound))[0]  # Low trials excluding close-to-median
    inds_high = np.where((pupil_trials > upper_bound))[0]  # High trials excluding close-to-median

    print(f"Excluding trials in range: {lower_bound} - {upper_bound}")
    print(f"Number of low trials: {len(inds_low)}")
    print(f"Number of high trials: {len(inds_high)}")

    # Process clusters and spike times for analysis
    pre_time = 0.15
    post_time = 0.15
    initial_time = 0.035

    results = []

    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]

        # Only process "stimulus 1"
        stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == 1]

        # Process data for low and high groups excluding close-to-median trials
        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[inds_high]['stimstart'], pre_time, post_time)

        baseline_low_firing_rate = np.mean(baseline_counts_low) / pre_time
        baseline_high_firing_rate = np.mean(baseline_counts_high) / pre_time
        evoked_low_firing_rate = np.mean(evoked_counts_low) / post_time
        evoked_high_firing_rate = np.mean(evoked_counts_high) / post_time

        # Fano Factor calculation
        FF_low = np.var(evoked_counts_low) / np.mean(evoked_counts_low) if np.mean(evoked_counts_low) > 0 else np.nan
        FF_high = np.var(evoked_counts_high) / np.mean(evoked_counts_high) if np.mean(evoked_counts_high) > 0 else np.nan

        # Append results
        results.append({
            'Dataset': dataset_num,
            'Cluster': c,
            'Stimulus': 1,
            'Penetration': penetration_date,
            'Monkey Name': monkey_name,
            'Baseline Low Firing Rate': baseline_low_firing_rate,
            'Baseline High Firing Rate': baseline_high_firing_rate,
            'Evoked Low Firing Rate': evoked_low_firing_rate,
            'Evoked High Firing Rate': evoked_high_firing_rate,
            'Fano Factor Low': FF_low,
            'Fano Factor High': FF_high
        })

    # Convert results to a DataFrame for this dataset and save
    results_df = pd.DataFrame(results)
    csv_filename = f'results_dataset_{dataset_num}.csv'
    csv_fullpath = os.path.join(results_dir, csv_filename)
    results_df.to_csv(csv_fullpath, index=False)

    all_results.append(results_df)

# Combine results across all datasets


combined_results_df = pd.concat(all_results, ignore_index=True)

# Create Population Bar Plot (aggregated across datasets)

# Calculate mean and standard error of evoked firing rates (Low vs. High)
mean_low = np.mean(combined_results_df['Evoked Low Firing Rate'])
mean_high = np.mean(combined_results_df['Evoked High Firing Rate'])
abs_firing = abs(mean_high-mean_low)


stderr_low = np.std(combined_results_df['Evoked Low Firing Rate']) / np.sqrt(len(combined_results_df))
stderr_high = np.std(combined_results_df['Evoked High Firing Rate']) / np.sqrt(len(combined_results_df))

# Paired t-test to compare Low vs. High evoked firing rates
_, p_val = ttest_rel(combined_results_df['Evoked Low Firing Rate'], combined_results_df['Evoked High Firing Rate'])

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar('Low', mean_low, yerr=stderr_low, color='gray', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_high, yerr=stderr_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)

# Annotate p-value
y_max = max(mean_low + stderr_low, mean_high + stderr_high)  # Maximum height of the bar + error
line_y = y_max + 2  # Adjust the height of the line

# Plot horizontal line and annotate p-value
plt.plot([0, 1], [line_y, line_y], color='black', lw=1.5)  # Horizontal line between bars
plt.text(0.5, line_y + 0.2, f'p = {p_val:.2e}', ha='center', fontsize=12, fontweight='bold')
plt.text(0.7 ,line_y + 0.3,"Low Pupil:2.38, High Pupil:2.54 mm", fontsize=10, fontweight='bold')
# Set labels and format the plot
plt.xlabel('Pupil Diameter', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Formatting the axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Save the bar plot
barplot_filename = 'population_barplot.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Create Fano Factor Scatter Plot: Fano Factor High vs Low
plt.figure(figsize=(8, 6))
plt.scatter(combined_results_df['Fano Factor Low'], combined_results_df['Fano Factor High'], color='green', s=15)
plt.plot([0.1, max(combined_results_df['Fano Factor Low'])],
         [0.1, max(combined_results_df['Fano Factor Low'])], 'r-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Fano Factor Low', fontsize=14)
plt.ylabel('Fano Factor High', fontsize=14)

# Set axis limits to start from 0.1
plt.xlim(0.1, max(combined_results_df['Fano Factor Low']))
plt.ylim(0.1, max(combined_results_df['Fano Factor High']))
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

# Save and display the scatter plot
scatterplot_filename = 'population_fano_factor_scatter_plot.svg'
plt.savefig(os.path.join(results_dir, scatterplot_filename))
plt.show()

# Bar Plot: Mean Fano Factor High vs Low
plt.figure(figsize=(8, 6))
mean_ff_low = np.nanmean(combined_results_df['Fano Factor Low'])
mean_ff_high = np.nanmean(combined_results_df['Fano Factor High'])
sem_ff_low = sem(combined_results_df['Fano Factor Low'], nan_policy='omit')
sem_ff_high = sem(combined_results_df['Fano Factor High'], nan_policy='omit')

plt.bar('Low', mean_ff_low, yerr=sem_ff_low, color='gray', capsize=5, edgecolor='black', linewidth=3)  # Increase the linewidth
plt.bar('High', mean_ff_high, yerr=sem_ff_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)  # Increase the linewidth

plt.xlabel('', fontsize=14)
plt.ylabel('Fano Factor', fontsize=14)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

# Save the bar plot
barplot_filename = 'population_mean_fano_factor_bar_plot.svg'
plt.savefig(os.path.join(results_dir, barplot_filename))
plt.show() 


# Create Population Scatter Plot (aggregated across datasets)

plt.figure(figsize=(8, 8))

low_population_means = combined_results_df['Evoked Low Firing Rate']
high_population_means = combined_results_df['Evoked High Firing Rate']


n = len(low_population_means)
# Create the scatter plot
plt.scatter(low_population_means, high_population_means, color='black', s=30, label=f'n = {n}')

# Add a diagonal reference line
max_limit = max(low_population_means.max(), high_population_means.max()) + 10
plt.plot([1, max_limit], [1, max_limit], 'r-', linewidth=2)

# Set the x and y scales to log
plt.xscale('log')
plt.yscale('log')

# Set the limits for the x and y axes
plt.xlim(1, max_limit)
plt.ylim(1, max_limit)

# Aspect ratio and axis formatting
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

# Set the labels and add penetration date to the scatter plot
plt.xlabel('Mean Firing Rate: Low', fontsize=16, fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=12, frameon=False)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# Save and display the scatter plot
scatterplot_filename = 'Population_scatterplot.svg' 
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()




from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Prepare the dataset for decoding analysis
# Use 'Evoked Low Firing Rate' and 'Evoked High Firing Rate' as features
features = combined_results_df[['Evoked Low Firing Rate', 'Evoked High Firing Rate',
                                'Fano Factor Low', 'Fano Factor High']].fillna(0)
labels = (combined_results_df['Evoked High Firing Rate'] > combined_results_df['Evoked Low Firing Rate']).astype(int)  # Binary label: 0 = Low, 1 = High

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.3, random_state=42, stratify=labels)

# Initialize classifiers
log_reg = LogisticRegression(random_state=42)
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)

# Train and evaluate Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)
log_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

print("Logistic Regression Performance:")
print(f"Accuracy: {log_accuracy:.2f}, AUC: {log_auc:.2f}")
print(classification_report(y_test, y_pred_log))

# Train and evaluate Random Forest
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])


# Cross-validation for Logistic Regression
log_reg_cv_scores = cross_val_score(log_reg, features_scaled, labels, cv=5, scoring='roc_auc')
print(f"Logistic Regression CV AUC: {log_reg_cv_scores.mean():.2f} ± {log_reg_cv_scores.std():.2f}")

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_clf, features_scaled, labels, cv=5, scoring='roc_auc')
print(f"Random Forest CV AUC: {rf_cv_scores.mean():.2f} ± {rf_cv_scores.std():.2f}")

print("Random Forest Performance:")
print(f"Accuracy: {rf_accuracy:.2f}, AUC: {rf_auc:.2f}")
print(classification_report(y_test, y_pred_rf))

# Visualize feature importance from Random Forest
import matplotlib.pyplot as plt
feature_importances = rf_clf.feature_importances_
plt.figure(figsize=(8, 6))
plt.bar(features.columns, feature_importances, color='skyblue', edgecolor='black')
plt.ylabel('Importance', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
plt.colorbar()
plt.xticks([0, 1], ['Low', 'High'], fontsize=12)
plt.yticks([0, 1], ['Low', 'High'], fontsize=12)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()










import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.io import loadmat
from utils import utils
from lib import readSGLX
import os

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Constants for penetration date and monkey name
penetration_date = ''
monkey_name = ''

# Threshold for high mean firing rate comparison
FIRING_RATE_THRESHOLD = 5

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

# Store results from all datasets
all_results = []
unique_clusters = set()  # Set to track unique clusters

# Loop through dataset(s)
for dataset_num in range(1, 2):  # Loop through the desired dataset(s)
    print(f"Processing dataset {dataset_num}...")

    # Load data for each dataset
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])

    # Load spike times and cluster data
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)

    # Load the trial mat file and extract stimuli
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']

    def linear_stimuli(expt_info):
        stims = np.array([])  # H
        for i in range(expt_info.trial_records.shape[0]):
            stims = np.hstack((stims, expt_info.trial_records[i].trImage))
        return stims

    stimuli = linear_stimuli(expt_info)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]  # Removing NaN values from stimuli

    trialDF = pd.read_csv(trial_DF_path)
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)

    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

    # Calculate median pupil diameter
    def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
        medians = []
        for index, row in stimulusDF.iterrows():
            start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
            stop_sample = int(row['stimstop'] * sample_rate)
            trial_data = pupil_diameter[start_sample:stop_sample]
            median_diameter = np.median(trial_data)
            medians.append(median_diameter)
        return np.array(medians)

    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Process clusters and spike times for analysis
    pre_time = 0.15
    post_time = 0.15
    initial_time = 0.035

    results = []

    # Loop through clusters
    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]
        
        # Loop through stimuli
        for stimulus_num in stimulusDF['stimuli'].unique():
            stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == stimulus_num]
            
            # Calculate the mean evoked firing rate for the current stimulus
            baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimDF_tmp["stimstart"].values, pre_time, post_time, initial_time)
            baseline_rate_mean = np.mean(baseline_rate_count) / pre_time
            evoked_rate_mean = np.mean(evoked_rate_count) / post_time
            
            # Select only stimuli where evoked rate is greater than a threshold
            if evoked_rate_mean >= baseline_rate_mean + FIRING_RATE_THRESHOLD:
                # Add this cluster to the set of unique clusters
                unique_clusters.add(c)

                # Continue with statistical analysis
                valid_inds_low = [i for i in inds_low if i < len(stimDF_tmp)]
                valid_inds_high = [i for i in inds_high if i < len(stimDF_tmp)]

                baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_low]['stimstart'], pre_time, post_time)
                baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_high]['stimstart'], pre_time, post_time)

                baseline_high_firing_rate = np.mean(baseline_counts_high) / pre_time
                baseline_low_firing_rate = np.mean(baseline_counts_low) / pre_time
                evoked_high_firing_rate = np.mean(evoked_counts_high) / post_time
                evoked_low_firing_rate = np.mean(evoked_counts_low) / post_time

                AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

                t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
                t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

                baseline_class = 'no effect'
                evoked_class = 'no effect'
                if p_val_baseline < 0.05:
                    baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
                if p_val_evoked < 0.05:
                    evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

                # Fano Factor calculation for Low and High trials
                FF_low = np.var(evoked_counts_low) / np.mean(evoked_counts_low) if np.mean(evoked_counts_low) > 0 else np.nan
                FF_high = np.var(evoked_counts_high) / np.mean(evoked_counts_high) if np.mean(evoked_counts_high) > 0 else np.nan

                # Save results for each cluster-stimulus combination
                results.append({
                    'Dataset': dataset_num,
                    'Cluster': c,
                    'Stimulus': stimulus_num,
                    'Penetration': penetration_date,
                    'Monkey Name': monkey_name,
                    'Effect Size': AMI,
                    'Baseline Classification': baseline_class,
                    'Evoked Classification': evoked_class,
                    'Baseline p-value': p_val_baseline,
                    'Evoked p-value': p_val_evoked,
                    'Baseline High Firing Rate': baseline_high_firing_rate,
                    'Baseline Low Firing Rate': baseline_low_firing_rate,
                    'Evoked High Firing Rate': evoked_high_firing_rate,
                    'Evoked Low Firing Rate': evoked_low_firing_rate,
                    'Fano Factor Low': FF_low,
                    'Fano Factor High': FF_high
                })

    # Convert results to a DataFrame for this dataset and save
    results_df = pd.DataFrame(results)
    csv_filename = f'2024-04-17_Population_classification_results_dataset_{dataset_num}.csv'
    csv_fullpath = os.path.join(results_dir, csv_filename)
    results_df.to_csv(csv_fullpath, index=False)

    all_results.append(results_df)

# Combine results across all datasets
combined_results_df = pd.concat(all_results, ignore_index=True)

# Calculate mean and standard error for low and high evoked firing rates
mean_low = np.mean(combined_results_df['Evoked Low Firing Rate'])
mean_high = np.mean(combined_results_df['Evoked High Firing Rate'])
stderr_low = np.std(combined_results_df['Evoked Low Firing Rate']) / np.sqrt(len(combined_results_df))
stderr_high = np.std(combined_results_df['Evoked High Firing Rate']) / np.sqrt(len(combined_results_df))

#
_, p_val = ttest_rel(combined_results_df['Evoked Low Firing Rate'], combined_results_df['Evoked High Firing Rate'])

# Create Population Bar Plot
plt.figure(figsize=(10, 6))
plt.bar('Low', mean_low, yerr=stderr_low, color='grey', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_high, yerr=stderr_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)

# Annotate p-value
y_max = max(mean_low + stderr_low, mean_high + stderr_high)
line_y = y_max + 2
plt.plot([0, 1], [line_y, line_y], color='black', lw=1.5)  # Line between bars
plt.text(0.5, line_y + 0.2, f'p = {p_val:.2e}', ha='center', fontsize=12, fontweight='bold')

plt.xlabel('Pupil Diameter', fontsize=16, fontweight='bold')
plt.ylabel('Firing Rate (Hz)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

barplot_filename = '_population_barplot.svg' 
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()

# Create Population Scatter Plot
plt.figure(figsize=(8, 8))
low_population_means = combined_results_df['Evoked Low Firing Rate']
high_population_means = combined_results_df['Evoked High Firing Rate']

# Create scatter plot
plt.scatter(low_population_means, high_population_means, color='black', s=30, label=f'n = {len(unique_clusters)}')

# Add diagonal reference line
max_limit = max(low_population_means.max(), high_population_means.max()) + 10
plt.plot([1, max_limit], [1, max_limit], 'r-', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlim(1, max_limit)
plt.ylim(1, max_limit)

plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

plt.xlabel('Mean Firing Rate: Low', fontsize=16, fontweight='bold')
plt.ylabel('Mean Firing Rate: High', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', fontsize=12, frameon=False)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

scatterplot_filename = '_Population_scatterplot.svg' 
plt.savefig(os.path.join(results_dir, scatterplot_filename))
plt.show()

# Create Fano Factor Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(combined_results_df['Fano Factor Low'], combined_results_df['Fano Factor High'], color='green', s=15)
plt.plot([0.1, max(combined_results_df['Fano Factor Low'])],
         [0.1, max(combined_results_df['Fano Factor Low'])], 'r-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Fano Factor Low', fontsize=14)
plt.ylabel('Fano Factor High', fontsize=14)

plt.xlim(0.1, max(combined_results_df['Fano Factor Low']))
plt.ylim(0.1, max(combined_results_df['Fano Factor High']))
plt.gca().set_aspect('equal', adjustable='box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(4)

scatterplot_filename = 'population_fano_factor_scatter_plot.svg'
plt.savefig(os.path.join(results_dir, scatterplot_filename))
plt.show()

# Bar Plot: Mean Fano Factor High vs Low
plt.figure(figsize=(8, 6))
mean_ff_low = np.nanmean(combined_results_df['Fano Factor Low'])
mean_ff_high = np.nanmean(combined_results_df['Fano Factor High'])
sem_ff_low = sem(combined_results_df['Fano Factor Low'], nan_policy='omit')
sem_ff_high = sem(combined_results_df['Fano Factor High'], nan_policy='omit')

plt.bar('Low', mean_ff_low, yerr=sem_ff_low, color='gray', capsize=5, edgecolor='black', linewidth=3)
plt.bar('High', mean_ff_high, yerr=sem_ff_high, color='#FF0000', capsize=5, edgecolor='black', linewidth=3)

plt.xlabel('Pupil Diameter', fontsize=14)
plt.ylabel('Fano Factor', fontsize=14)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

barplot_filename = 'population_mean_fano_factor_bar_plot.svg'
plt.savefig(os.path.join(results_dir, barplot_filename))
plt.show()










import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.io import loadmat
from scipy.signal import detrend
from utils import utils
from lib import readSGLX
import os
import data_analysislib as dalib  # Assuming this has simple_mean_match

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Constants for penetration date and monkey name
penetration_date = '2024'
monkey_name = 'WJ'
FIRING_RATE_THRESHOLD = 5  # Threshold for high mean firing rate comparison

# Function to calculate spike counts for baseline and evoked
def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    return np.array(baseline_counts), np.array(evoked_counts)

# Load all datasets
all_results = []
unique_clusters = set()  # Track unique clusters

# Loop over datasets
for dataset_num in range(1, 2):
    print(f"Processing dataset {dataset_num}...")
    
    # Load data paths
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    
    # Load datasets
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']

    # Corrected linear_stimuli function
    def linear_stimuli(expt_info):
        stims = []
        for rec in expt_info.trial_records:
            # Check if trImage exists and is a single number (not a sequence)
            if hasattr(rec, 'trImage') and np.isscalar(rec.trImage):
                stims.append(rec.trImage)
        return np.array(stims)

    # Use the updated function
    stimuli = linear_stimuli(expt_info)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]

    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend((pupil_diameter - np.nanmin(pupil_diameter)) / np.nanmax(pupil_diameter))

    # Median pupil diameter calculation
    def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
        medians = []
        for _, row in stimulusDF.iterrows():
            start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
            stop_sample = int(row['stimstop'] * sample_rate)
            trial_data = pupil_diameter[start_sample:stop_sample]
            medians.append(np.median(trial_data))
        return np.array(medians)

    # Ensure sRate is defined
    sRate = readSGLX.SampRate(meta)  # Assuming `meta` is already loaded
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Initialize arrays for storing Fano Factors
    spkC_low, spkC_high, FF_low, FF_high = [], [], [], []

    # Assuming spike_times_clusters is defined
    for c, Y in spike_times_clusters.items():
        spkC, BSL = get_spike_counts(Y, stimulusDF['stimstart'], 0.15, 0.15, 0.035)
        res = ttest_rel(spkC, BSL)
        if np.mean(spkC) > np.mean(BSL) + FIRING_RATE_THRESHOLD:
            unique_clusters.add(c)
            spkC_zscored = stats.zscore(spkC)
            
            # Calculate Fano Factors for low and high conditions
            FF_low_tmp, FF_high_tmp = [], []
            spkC_low_tmp, spkC_high_tmp = [], []
            for stim_num in np.unique(stimuli):
                inds = np.where(stimuli == stim_num)[0]
                inds_low_tmp = np.intersect1d(inds, inds_low)
                inds_high_tmp = np.intersect1d(inds, inds_high)
                FF_low_tmp.append(np.var(spkC[inds_low_tmp]) / np.mean(spkC[inds_low_tmp]))
                FF_high_tmp.append(np.var(spkC[inds_high_tmp]) / np.mean(spkC[inds_high_tmp]))
                spkC_low_tmp.append(np.mean(spkC[inds_low_tmp]))
                spkC_high_tmp.append(np.mean(spkC[inds_high_tmp]))
            FF_low.append(np.mean(FF_low_tmp))
            FF_high.append(np.mean(FF_high_tmp))
            spkC_low.append(np.mean(spkC_low_tmp))
            spkC_high.append(np.mean(spkC_high_tmp))

    spkC_low, spkC_high = np.array(spkC_low), np.array(spkC_high)
    FF_low, FF_high = np.array(FF_low), np.array(FF_high)

    # Mean Matching for Fano Factors
    count_bins = np.arange(0, 301, 1)
    mean_low_matched, mean_high_matched, FF_low_matched, FF_high_matched = dalib.simple_mean_match(
        spkC_low, spkC_high, FF_low, FF_high, count_bins)

    # Plot mean-matched Fano factors
    plt.figure(figsize=(12, 6))

    # Fano Factor scatter plot
    plt.subplot(1, 2, 1)
    plt.plot(FF_low, FF_high, 'ko', markersize=3, markerfacecolor='None')
    plt.plot([0.5, 8], [0.5, 8], 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fano Factor Low')
    plt.ylabel('Fano Factor High')
    plt.title('Fano Factor Scatter Plot')

    # Matched Fano Factor bar plot
    plt.subplot(1, 2, 2)
    FF_low_mean_matched = np.mean(FF_low_matched)
    FF_high_mean_matched = np.mean(FF_high_matched)
    FF_low_matched_SE = np.std(FF_low_matched) / np.sqrt(len(FF_low_matched))
    FF_high_matched_SE = np.std(FF_high_matched) / np.sqrt(len(FF_high_matched))
    plt.bar([1, 2], [FF_low_mean_matched, FF_high_mean_matched], yerr=[FF_low_matched_SE, FF_high_matched_SE],
            color=['gray', 'black'], capsize=5)
    plt.xticks([1, 2], ['Low', 'High'])
    plt.ylabel('Fano Factor (Mean Matched)')
    plt.title('Mean Matched Fano Factor Comparison')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mean_matched_fano_factors.svg'))
    plt.show()


