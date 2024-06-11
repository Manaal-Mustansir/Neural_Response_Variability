from utils import utils
from lib import readSGLX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import detrend

# Load files
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
trial_DF_path = utils.getFilePath(windowTitle="trials_DF_Stimulus", filetypes=[('Trials_Stimulus csv file', '*.csv')])
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data",filetypes=[("Numpy array","*.npy")])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
trialDF = pd.read_csv(trial_DF_path)
pupil_diameter = np.load(pupilFullPath)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file",filetypes=[("NIdq binary","*.bin")]) # type: ignore
meta  = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)

# Get spike times for each cluster
spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

def meanvar_PSTH(data,count_window=100,style='same',return_bootdstrs=False,nboots=1000):
    data = data > 0
    if style == 'valid':
        mean_timevector = np.nan*np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan*np.ones((data.shape[1] - count_window + 1))
        tmp  = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan*np.ones((data.shape[1]))
        vari_timevector = np.nan*np.ones((data.shape[1]))
        tmp  = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        # compute spike counts in sliding window
        tmp[i,:] = np.convolve(data[i,:],np.ones(count_window,),style)
            
    vari_timevector = np.var(tmp,axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0],(tmp.shape[0],nboots))
        mean_timevector_booted = np.nan * np.ones((nboots,tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots,tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:],axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:],axis=0)

    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    #
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted

""" def plot_psth(data_matrix, time_bins, cluster_index, mean_firing_rate):
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot PSTH for each trial as event plot
    for trial_idx in range(data_matrix.shape[0]):
        spike_times = time_bins[data_matrix[trial_idx, :] > 0]
        ax.eventplot(spike_times, lineoffsets=trial_idx + 1, colors='black')
    
    ax.set_title(f'PSTH - Cluster {cluster_index} (Mean Firing Rate: {mean_firing_rate:.2f} spikes/sec)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    
    plt.tight_layout()
    plt.show() """

def return_psth(Y, trialDF, bin_size=0.001, pre_window=0.150, post_window=0.150):
    
    num_bins = int((pre_window + post_window) / bin_size) 
    data_matrix = np.zeros((len(trialDF), num_bins))    

    for trial_idx, row in trialDF.iterrows():
        stimstart = row['stimstart']

        # Define time window around stimstart
        start_time = stimstart - pre_window
        stop_time = stimstart + post_window

        # Filter spike times within the trial window
        spikes_in_window = Y[(Y >= start_time) & (Y <= stop_time)]

        # Bin the spikes
        bin_counts, _ = np.histogram(spikes_in_window, bins=np.linspace(start_time, stop_time, num_bins+1))
        data_matrix[trial_idx, :] = bin_counts

    
    mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted = meanvar_PSTH(data_matrix,count_window=100,style='same',return_bootdstrs=True,nboots=1000)
    t = np.linspace(start_time, stop_time, num_bins)
    return mean_timevector, vari_timevector, mean_timevector_booted, vari_timevector_booted,t

def get_spike_counts(Y, stimulusDF):
    spkC_evk = np.NaN * np.ones(len(stimulusDF))
    spkC_bsl = np.NaN * np.ones(len(stimulusDF))
                    
    for index, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        stop_time = row['stimstop']
        spikes_in_trial = Y[(Y >= start_time + 0.035) & (Y <= stop_time)]
        spikes_in_bsl = Y[(Y >= start_time - 0.15 + 0.035) & (Y <= start_time)]
        spike_count_trial = len(spikes_in_trial)
        spike_count_bsl = len(spikes_in_bsl)
        spkC_evk[index] = spike_count_trial
        spkC_bsl[index] = spike_count_bsl

    return spkC_evk, spkC_bsl

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample  = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)


def get_pupil_Diameter(stimulusDF, pupil_diameter,sRate):

    # ad hoc cleaning needed for 2024-04-11 data
    #pupil_diameter[0:80000] = -3900.0
    pupil_diameter = detrend(pupil_diameter)
    pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
    pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
    pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)
    
    #inds_low = np.where((pupil_trials > 0.25) & (pupil_trials < 0.35))[0]
    pup_median = np.median(pupil_trials)
    inds_low = np.where(pupil_trials < pup_median)[0]
    inds_high = np.where(pupil_trials > pup_median)[0]

    return inds_low, inds_high


good_clusters = []
spkC = {}
BSL = {}
spkC_zscored = {}

inds_low, inds_high = get_pupil_Diameter(trialDF, pupil_diameter,sRate)

for i in spike_times_clusters.keys():
    Y = spike_times_clusters[i]    

    spkC[i],BSL[i] = get_spike_counts(Y, trialDF)    
    if np.mean(spkC[i]) > np.mean(BSL[i]) + 3:
        good_clusters.append(i)
        # Call with trialDF_low
        mean_timevector, vari_timevector, mean_timevector_booted, vari_timevector_booted,t = return_psth(Y, trialDF)
        # Call with trialDF_high
        #mean_timevector, vari_timevector, mean_timevector_booted, vari_timevector_booted,t = return_psth(Y, trialDF)

        fig,ax = plt.subplots(2,1)
        t = np.linspace(-150,150,300)
        psth_low  = mean_timevector - np.std(mean_timevector_booted,axis=0)
        psth_high = mean_timevector + np.std(mean_timevector_booted,axis=0)
        ax[0].fill_between(t,psth_low,psth_high,color='grey')
        ax[0].plot(t,mean_timevector,'k--')        
        #ax[1].plot(t,fano_factor,'r-')






