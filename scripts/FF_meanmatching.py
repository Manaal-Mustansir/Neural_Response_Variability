import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel, sem
from scipy.io import loadmat
from scipy.signal import detrend
from utils import utils
from lib import readSGLX
import os
import statsmodels.api as sm

# Results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

csv_file_path = utils.getFilePath(
    windowTitle="Select CSV File",
    filetypes=[("CSV files", "*.csv")]
)

# Load the selected CSV file
df = pd.read_csv(csv_file_path)
# Constants for penetration date and monkey name
penetration_date = '2024'
monkey_name = 'Sansa'
FIRING_RATE_THRESHOLD = 5


def mean_match(mean1,mean2,variance1,variance2,count_bins,nboots=1000):
    import statsmodels.api as sm
    # the algorithm brakes with nans
    variance1 = variance1[~np.isnan(mean1)]
    mean1 = mean1[~np.isnan(mean1)]
    variance2 = variance2[~np.isnan(mean2)]
    mean2 = mean2[~np.isnan(mean2)]
    
    bin_mems1 = np.digitize(mean1,count_bins)
    bin_mems2 = np.digitize(mean2,count_bins)
    slope1 = np.empty(0)
    slope2 = np.empty(0)
    for boot in range(nboots):
        inds2remove1 = np.empty(0)
        inds2remove2 = np.empty(0)
        # arange uses half open interval [start,stop)
        for i in np.arange(np.min(count_bins),np.max(count_bins)+1,1):
            n1 = np.sum(bin_mems1 == i)
            n2 = np.sum(bin_mems2 == i)

            if n1 > n2:
                inds = np.where(bin_mems1 == i)[0]
                inds2remove1 = np.append(inds2remove1,np.random.choice(inds,n1-n2,replace=False))
            elif n2 > n1:
                inds = np.where(bin_mems2 == i)[0]
                inds2remove2 = np.append(inds2remove2,np.random.choice(inds,n2-n1,replace=False))
                                     
        meantmp1 = np.delete(mean1, inds2remove1.astype(int))
        varstmp1 = np.delete(variance1, inds2remove1.astype(int))
        meantmp2 = np.delete(mean2, inds2remove2.astype(int))
        varstmp2 = np.delete(variance2, inds2remove2.astype(int))


        model_1  = sm.OLS(varstmp1, meantmp1)
        results1 = model_1.fit()
        model_2  = sm.OLS(varstmp2, meantmp2)
        results2 = model_2.fit()

        slope1 = np.append(slope1,results1.params[0])
        slope2 = np.append(slope2,results2.params[0])
        
    return slope1,slope2,meantmp1,meantmp2


# Spike counting function for baseline and evoked responses
def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts, evoked_counts = [], []
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    return np.array(baseline_counts), np.array(evoked_counts)

# Load datasets
for dataset_num in range(1, 2):
    print(f"Processing dataset {dataset_num}...")
    
    # Load data paths
    spike_times_secpath = utils.getFilePath(windowTitle=f"Spike_times Dataset {dataset_num}", filetypes=[('Spike times numpy file', '*.npy')])
    clusters_path = utils.getFilePath(windowTitle=f"Select_clusters Dataset {dataset_num}", filetypes=[('Clusters numpy file', '*.npy')])
    trial_metadata_path = utils.getFilePath(windowTitle=f"Metadata Dataset {dataset_num}", filetypes=[('Mat-file', '*.mat')])
    stimulus_DF_path = utils.getFilePath(windowTitle=f"stimstart Dataset {dataset_num}", filetypes=[('stimulus csv file', '*.csv')])
    trial_DF_path = utils.getFilePath(windowTitle=f"trialstart Dataset {dataset_num}", filetypes=[('trial csv file', '*.csv')])
    
    spike_times_sec = np.load(spike_times_secpath)
    clusters = np.load(clusters_path)
    stimulusDF = pd.read_csv(stimulus_DF_path)
    trialDF = pd.read_csv(trial_DF_path)
    mat = loadmat(trial_metadata_path, struct_as_record=False, squeeze_me=True)
    expt_info = mat['expt_info']
    
    # Extract and clean stimuli
    def linear_stimuli(expt_info):
        stims = np.array([])  
        for i in range(expt_info.trial_records.shape[0]):
            stims = np.hstack((stims, expt_info.trial_records[i].trImage))
        return stims

    stimuli = linear_stimuli(expt_info)
    stimulusDF['stimuli'] = stimuli[~np.isnan(stimuli)]

    # Load pupil diameter data
    binFullPath = utils.getFilePath(windowTitle=f"Binary nidq file Dataset {dataset_num}", filetypes=[("NIdq binary", "*.bin")])
    meta = readSGLX.readMeta(binFullPath)
    sRate = readSGLX.SampRate(meta)
    pupilFullPath = utils.getFilePath(windowTitle=f"Pupil diameter data Dataset {dataset_num}", filetypes=[("Numpy array", '*.npy')])
    pupil_diameter = np.load(pupilFullPath)
    pupil_diameter = detrend((pupil_diameter - np.nanmin(pupil_diameter)) / np.nanmax(pupil_diameter))

    # Calculate median pupil diameter
    def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
        medians = []
        for _, row in stimulusDF.iterrows():
            start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
            stop_sample = int(row['stimstop'] * sample_rate)
            trial_data = pupil_diameter[start_sample:stop_sample]
            medians.append(np.median(trial_data))
        return np.array(medians)

    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    # Get spike times clustered
    spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)
    spkC_low, spkC_high, FF_low, FF_high, FR_low, FR_high = [], [], [], [], [], []
    pre_time = 0.15
    post_time = 0.15
    initial_time = 0.035
    for c in spike_times_clusters.keys():
        Y = spike_times_clusters[c]
        
        for stimulus_num in stimulusDF['stimuli'].unique():
            stimDF_tmp = stimulusDF[stimulusDF['stimuli'] == stimulus_num]
            baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimDF_tmp["stimstart"].values, pre_time=0.15, post_time=0.15, initial_time=0.035)
            baseline_rate_mean = np.mean(baseline_rate_count) / 0.15
            evoked_rate_mean = np.mean(evoked_rate_count) / 0.15
            
            if evoked_rate_mean >= baseline_rate_mean + FIRING_RATE_THRESHOLD:
                valid_inds_low = [i for i in inds_low if i < len(stimDF_tmp)]
                valid_inds_high = [i for i in inds_high if i < len(stimDF_tmp)]

                baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_low]['stimstart'], 0.15, 0.15)
                baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimDF_tmp.iloc[valid_inds_high]['stimstart'], 0.15, 0.15)

                FF_low.append(np.var(evoked_counts_low) / np.mean(evoked_counts_low) if np.mean(evoked_counts_low) > 0 else np.nan)
                FF_high.append(np.var(evoked_counts_high) / np.mean(evoked_counts_high) if np.mean(evoked_counts_high) > 0 else np.nan)
                spkC_low.append(np.mean(evoked_counts_low))
                FR_low.append(np.mean(evoked_counts_low) / post_time)
                FR_high.append(np.mean(evoked_counts_high) / post_time)
                spkC_high.append(np.mean(evoked_counts_high))


    # Mean Matching for Fano Factors
    spkC_low, spkC_high, FF_low, FF_high = np.array(spkC_low), np.array(spkC_high), np.array(FF_low), np.array(FF_high)
    count_bins = np.arange(0, 301, 1)
    slope1, slope2, meantmp1, meantmp2 = mean_match(spkC_low, spkC_high, FF_low, FF_high, count_bins)
    
    # Plot  Fano factors
    plt.figure(figsize=(18, 6))

    # Fano Factor scatter plot
    plt.subplot(1, 5, 1)
    plt.plot(FF_low, FF_high, 'ko', markersize=3, markerfacecolor='None')
    plt.plot([0.5, 8], [0.5, 8], 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fano Factor Low')
    plt.ylabel('Fano Factor High')
   
    # Matched Fano Factor bar plot
    plt.subplot(1, 5, 2)
    # Get the number of units for each condition
    n_low = len(FF_low)  # Count of values n
    n_high = len(FF_high)
    
    FF_low_matched = np.array(FF_low)[meantmp1.astype(int)]
    FF_high_matched = np.array(FF_high)[meantmp2.astype(int)]
    
    n_matched_low = len(FF_low_matched)  # Number of units in matched condition
    n_matched_high = len(FF_high_matched)
    # Apply 
    
    # matched indices
    FF_low_mean_matched, FF_high_mean_matched = np.nanmean(FF_low_matched), np.nanmean(FF_high_matched)
    FF_low_matched_SE, FF_high_matched_SE = sem(FF_low_matched), sem(FF_high_matched)


    plt.bar([1, 2], [FF_low_mean_matched, FF_high_mean_matched], yerr=[FF_low_matched_SE, FF_high_matched_SE], color=['gray', 'red'], capsize=5)
    plt.xticks([1, 2], ['Low', 'High'])
    plt.ylabel('Mean-matched Fano Factor')
    # Display `n` values above bars for Mean-Matched
    plt.text(1, FF_low_mean_matched + 0.2, f"n={n_matched_low}", ha='center', fontsize=8, fontweight='bold')
    plt.text(2, FF_high_mean_matched + 0.2, f"n={n_matched_high}", ha='center', fontsize=8, fontweight='bold')
    print("Original FF_low:", FF_low)
    print("Original FF_high:", FF_high)
    print("Matched FF_low:", FF_low_mean_matched)
    print("Matched FF_high:", FF_high_mean_matched)
    
    FF_low_mean = np.nanmean(FF_low)
    FF_high_mean = np.nanmean(FF_high)
    FF_low_SE = sem(FF_low, nan_policy='omit')
    FF_high_SE = sem(FF_high, nan_policy='omit')

   # Plot Raw Fano Factor
    plt.subplot(1, 5, 3)
    plt.bar([1, 2], [FF_low_mean, FF_high_mean], yerr=[FF_low_SE, FF_high_SE], color=['gray', 'red'], capsize=5)
    plt.xticks([1, 2], ['Low', 'High'])
    plt.ylabel('Fano Factor Raw')
    plt.text(1, FF_low_mean + 0.1, f"n={int(n_low)}", ha='center', fontsize=6, fontweight='bold')
    plt.text(2, FF_high_mean + 0.1, f"n={int(n_high)}", ha='center', fontsize=6, fontweight='bold')
    
    # Plot for firing rates
    mean_low = np.nanmean(df["Evoked Low Firing Rate"])
    mean_high = np.nanmean(df["Evoked High Firing Rate"])
    stderr_low = sem(df["Evoked Low Firing Rate"], nan_policy='omit')
    stderr_high = sem(df["Evoked High Firing Rate"], nan_policy='omit')
    plt.subplot(1, 5, 4)
    plt.bar(['Low', 'High'], [mean_low, mean_high], 
        yerr=[stderr_low, stderr_high], 
        color=['gray', '#FF0000'], capsize=5, edgecolor='black', linewidth=3)
    plt.ylabel('Firing Rate (Hz)')
    plt.xlabel('Pupil Diameter')




    # Plot for Mean-matched firing rates
    FR_low_matched = np.array(FR_low)[meantmp1.astype(int)]
    FR_high_matched = np.array(FR_high)[meantmp2.astype(int)]
    # matched indices
    FR_low_mean_matched, FR_high_mean_matched = np.nanmean(FR_low_matched), np.nanmean(FR_high_matched)
    FR_low_matched_SE, FR_high_matched_SE = sem(FR_low_matched), sem(FR_high_matched)
    n_matched_low_FR = len(FR_low_matched)  # Number of units in matched condition
    n_matched_high_FR = len(FR_high_matched)
    plt.subplot(1, 5, 5)
    
    plt.bar([1, 2], [FR_low_mean_matched, FR_high_mean_matched], yerr=[FR_low_matched_SE, FR_high_matched_SE], color=['gray', 'red'], capsize=5)
    plt.xticks([1, 2], ['Low', 'High'])
    plt.ylabel('Mean-matched Firing Rate')
    # Display `n` values above bars for Mean-Matched FR
    plt.text(1, FR_low_mean_matched + 0.2, f"n={n_matched_low_FR}", ha='center', fontsize=8, fontweight='bold')
    plt.text(2, FR_high_mean_matched + 0.2, f"n={n_matched_high_FR}", ha='center', fontsize=8, fontweight='bold')
    print("Matched FR_low:", FR_low_mean_matched)
    print("Matched FR_high:", FR_high_mean_matched)
    # Save and show plot
    plt.savefig(os.path.join(results_dir, 'Penetration-1..svg'))
    plt.show()

