import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind
from utils import utils
from lib import readSGLX
import os
import seaborn as sns
import matplotlib.patches as mpatches 

# results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

count_window = 20

def plot_raster(Y, ax, stimulusDF, title, pre_time, post_time, color):
    tr = 0
    for _, row in stimulusDF.iterrows():
        tr += 1
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        ax.eventplot(spikes_in_trial - start_time, color=color, linewidths=0.5, lineoffsets=tr)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_xlim(-pre_time, post_time)

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15, bin_size=0.001):
    trial_spike_data = []
    bin_edges = np.arange(-pre_time, post_time + bin_size, bin_size)
    expected_hist_length = len(bin_edges) - 1
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        
        if len(spikes_in_trial) == 0:
            hist = np.zeros(expected_hist_length)
        else:
            spike_times_relative = spikes_in_trial - start_time
            hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

 
def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000, binarize: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if binarize:
        data = data > 0
    
    if style == 'valid':
        mean_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        tmp = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan * np.ones((data.shape[1]))
        vari_timevector = np.nan * np.ones((data.shape[1]))
        tmp = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        tmp[i,:] = np.convolve(data[i,:], np.ones(count_window,), style)
            
    vari_timevector = np.var(tmp, axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0], (tmp.shape[0], nboots))
        mean_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:], axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:], axis=0)
    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Store spike data for AMI calculation
all_spike_data_low = []
all_spike_data_high = []

# Define baseline period 
pre_time  = 0.2
post_time = 0.2
initial_time=0.035

def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

results = []

# Define constants for penetration and monkey name
penetration = '2024-04-17'
monkey_name = 'Sansa'

for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    
    baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time, post_time, initial_time)
    baseline_rate_mean= np.mean(baseline_rate_count)/pre_time
    evoked_rate_mean = np.mean(evoked_rate_count)/post_time

    if evoked_rate_mean >=  baseline_rate_mean + 5:
              
        fig, axs = plt.subplots(3, 1, figsize=(10, 16))
        
        plot_raster(Y, 
                    axs[0], 
                    stimulusDF.iloc[inds_low], 
                    title=f'Cluster {c} - Low Pupil Diameter',
                    pre_time=0.2,                    
                    post_time=0.2,
                    color='black')
        
        plot_raster(Y, 
                    axs[1], 
                    stimulusDF.iloc[inds_high], 
                    title=f'Cluster {c} - High Pupil Diameter',
                    pre_time=0.2,
                    post_time=0.2,
                    color='red')
        
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF,pre_time,post_time)      

        valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
        valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

        spike_data_low = spike_data_all[valid_inds_low]
        spike_data_high = spike_data_all[valid_inds_high]
        
        mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
        mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window)

        total_spike_counts_low = np.sum(spike_data_low, axis=1)
        total_spike_counts_high = np.sum(spike_data_high, axis=1)
        
        baseline_low_firing_rate = np.mean(total_spike_counts_low)
        baseline_high_firing_rate = np.mean(total_spike_counts_high)

        fano_factor_low = var_psth_low / mean_psth_low
        fano_factor_high = var_psth_high / mean_psth_high

        bin_size = 0.001  # 1 ms bins
        time_vector = np.arange(-pre_time, post_time, bin_size)        

        stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
        stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

        normalizing_factor = count_window/1000.0

        axs[2].plot(time_vector, mean_psth_low/normalizing_factor, color='black', linestyle='--', label='Low Pupil Diameter')
        axs[2].plot(time_vector, mean_psth_high/normalizing_factor, color='red', label='High Pupil Diameter')

        axs[2].fill_between(time_vector, np.array(mean_psth_low - stderr_psth_low)/normalizing_factor, np.array(mean_psth_low + stderr_psth_low)/normalizing_factor, color='grey', alpha=0.3)
        axs[2].fill_between(time_vector, np.array(mean_psth_high - stderr_psth_high)/normalizing_factor, np.array(mean_psth_high + stderr_psth_high)/normalizing_factor, color='lightcoral', alpha=0.3)

        axs[2].set_xlim(-pre_time, post_time)  
        axs[2].set_title(f'PSTH for Cluster {c}')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Spike Rate (Hz)')
        axs[2].legend()

        plt.tight_layout()
        svg_filename = f'cluster_{c}.svg'
        svg_fullpath = os.path.join(results_dir, svg_filename)
        plt.savefig(svg_fullpath)
        plt.close(fig)
        

        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], pre_time, post_time)
        
        baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
        baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
        evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
        evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

        AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

        t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
        t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

        baseline_class = 'no effect'
        evoked_class = 'no effect'
        if p_val_baseline < 0.05:
            baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
        if p_val_evoked < 0.05:
            evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

        results.append({
            'Cluster': c,
            'Penetration': penetration,
            'Monkey Name': monkey_name,
            'Effect Size': AMI,
            'Layers': '',
            'Baseline Classification': baseline_class,
            'Evoked Classification': evoked_class,
            'Baseline p-value': p_val_baseline,
            'Evoked p-value': p_val_evoked,
            'Baseline High Firing Rate': baseline_high_firing_rate,
            'Baseline Low Firing Rate': baseline_low_firing_rate,
            'Evoked High Firing Rate': evoked_high_firing_rate,
            'Evoked Low Firing Rate': evoked_low_firing_rate
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
csv_filename = 'classification_results_2024-04-17.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
results_df.to_csv(csv_fullpath, index=False)


# Bar plot for population effect
mean_low_population = np.mean([res['Evoked Low Firing Rate'] for res in results])
mean_high_population = np.mean([res['Evoked High Firing Rate'] for res in results])
n_neurons = len([res['Evoked High Firing Rate'] for res in results])

# Standard errors
stderr_low_population = np.std([res['Evoked Low Firing Rate'] for res in results]) / np.sqrt(len(results))
stderr_high_population = np.std([res['Evoked High Firing Rate'] for res in results]) / np.sqrt(len(results))

plt.figure(figsize=(10, 6))

# Use matplotlib text properties instead of LaTeX formatting
labels = ['Low Arousal', 'High Arousal']
colors = ['black', 'red']
bars = plt.bar(labels, 
               [mean_low_population, mean_high_population], 
               yerr=[stderr_low_population, stderr_high_population], 
               color=colors,  
               capsize=5)

# Setting custom text properties for bar labels
bars[0].set_label(r'Low Arousal')
bars[1].set_label(r'High Arousal')

plt.title('p < 0.05')
penetration_patch = mpatches.Patch(color='none', label='Penetration: 04-17-2024')
plt.legend(handles=[penetration_patch], loc='upper right', fontsize=12, frameon=False)
barplot_filename = 'population_effect.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()


# Scatter plot for high vs low 
plt.figure(figsize=(10, 6))
low_population_means = [res['Evoked Low Firing Rate'] for res in results]
high_population_means = [res['Evoked High Firing Rate'] for res in results]

# Determine the limit based on the maximum value across both axes
max_limit = max(max(low_population_means), max(high_population_means)) + 10 
# Scatter plot with low arousal in black and high arousal in red
plt.scatter(low_population_means, high_population_means, color='red', label='_nolegend_')
plt.scatter(low_population_means, low_population_means, color='black',  label='_nolegend_')
plt.plot([0, max_limit], [0, max_limit], 'k--')

plt.xlim(0, max_limit)
plt.ylim(0, max_limit)
plt.title('Pupil Diameter Effect')
plt.xlabel('Mean Firing Rate:Low')
plt.ylabel('Mean Firing Rate:High')
n_value_patch = mpatches.Patch(color='none', label=f'n = {n_neurons}')
penetration_patch = mpatches.Patch(color='none', label='Penetration: 04-17-2024')
plt.legend(handles=[n_value_patch, penetration_patch], loc='upper left', fontsize=12, frameon=False)
scatterplot_filename = 'high_vs_low_arousal_Scatter_plot.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()



# Swarm plot 
plt.figure(figsize=(10, 6))
sns.swarmplot(x="Evoked Classification", y="Evoked High Firing Rate", data=results_df, color='red')
sns.swarmplot(x="Evoked Classification", y="Evoked Low Firing Rate", data=results_df, color='black')
plt.xticks(['up', 'down', 'no effect'])
plt.yscale('log')
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='High Arousal')
black_patch = plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Low Arousal')
plt.legend(handles=[red_patch, black_patch])
plt.title('Evoked Firing Rates')
swarmplot_filename = 'swarm_plot.svg'
swarmplot_fullpath = os.path.join(results_dir, swarmplot_filename)
plt.savefig(swarmplot_fullpath)
plt.show()









import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind
from utils import utils
from lib import readSGLX
import os
import seaborn as sns

# results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

count_window = 20

def plot_raster(Y, ax, stimulusDF, title, pre_time, post_time, color):
    tr = 0
    for _, row in stimulusDF.iterrows():
        tr += 1
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        ax.eventplot(spikes_in_trial - start_time, color=color, linewidths=0.5, lineoffsets=tr)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_xlim(-pre_time, post_time)

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15, bin_size=0.001):
    trial_spike_data = []
    bin_edges = np.arange(-pre_time, post_time + bin_size, bin_size)
    expected_hist_length = len(bin_edges) - 1
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        
        if len(spikes_in_trial) == 0:
            hist = np.zeros(expected_hist_length)
        else:
            spike_times_relative = spikes_in_trial - start_time
            hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

 
def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000, binarize: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if binarize:
        data = data > 0
    
    if style == 'valid':
        mean_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        tmp = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan * np.ones((data.shape[1]))
        vari_timevector = np.nan * np.ones((data.shape[1]))
        tmp = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        tmp[i,:] = np.convolve(data[i,:], np.ones(count_window,), style)
            
    vari_timevector = np.var(tmp, axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0], (tmp.shape[0], nboots))
        mean_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:], axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:], axis=0)
    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Store spike data for AMI calculation
all_spike_data_low = []
all_spike_data_high = []

# Define baseline period 
pre_time  = 0.2
post_time = 0.2
initial_time=0.035

def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

results = []

# Define constants for penetration and monkey name
penetration = '2024-04-17'
monkey_name = 'Sansa'

for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    
    baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time, post_time, initial_time)
    baseline_rate_mean= np.mean(baseline_rate_count)/pre_time
    evoked_rate_mean = np.mean(evoked_rate_count)/post_time

    if evoked_rate_mean >=  baseline_rate_mean + 5:
              
        fig, axs = plt.subplots(3, 1, figsize=(10, 16))
        
        plot_raster(Y, 
                    axs[0], 
                    stimulusDF.iloc[inds_low], 
                    title=f'Cluster {c} - Low Pupil Diameter',
                    pre_time=0.2,                    
                    post_time=0.2,
                    color='black')
        
        plot_raster(Y, 
                    axs[1], 
                    stimulusDF.iloc[inds_high], 
                    title=f'Cluster {c} - High Pupil Diameter',
                    pre_time=0.2,
                    post_time=0.2,
                    color='red')
        
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF,pre_time,post_time)      

        valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
        valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

        spike_data_low = spike_data_all[valid_inds_low]
        spike_data_high = spike_data_all[valid_inds_high]
        
        mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
        mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window)

        total_spike_counts_low = np.sum(spike_data_low, axis=1)
        total_spike_counts_high = np.sum(spike_data_high, axis=1)
        
        baseline_low_firing_rate = np.mean(total_spike_counts_low)
        baseline_high_firing_rate = np.mean(total_spike_counts_high)

        fano_factor_low = var_psth_low / mean_psth_low
        fano_factor_high = var_psth_high / mean_psth_high

        bin_size = 0.001  # 1 ms bins
        time_vector = np.arange(-pre_time, post_time, bin_size)        

        stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
        stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

        normalizing_factor = count_window/1000.0

        axs[2].plot(time_vector, mean_psth_low/normalizing_factor, color='black', linestyle='--', label='Low Pupil Diameter')
        axs[2].plot(time_vector, mean_psth_high/normalizing_factor, color='red', label='High Pupil Diameter')

        axs[2].fill_between(time_vector, np.array(mean_psth_low - stderr_psth_low)/normalizing_factor, np.array(mean_psth_low + stderr_psth_low)/normalizing_factor, color='grey', alpha=0.3)
        axs[2].fill_between(time_vector, np.array(mean_psth_high - stderr_psth_high)/normalizing_factor, np.array(mean_psth_high + stderr_psth_high)/normalizing_factor, color='lightcoral', alpha=0.3)

        axs[2].set_xlim(-pre_time, post_time)  
        axs[2].set_title(f'PSTH for Cluster {c}')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Spike Rate (Hz)')
        axs[2].legend()

        plt.tight_layout()
        svg_filename = f'cluster_{c}.svg'
        svg_fullpath = os.path.join(results_dir, svg_filename)
        plt.savefig(svg_fullpath)
        plt.close(fig)
        

        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], pre_time, post_time)
        
        baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
        baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
        evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
        evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

        AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

        t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
        t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

        baseline_class = 'no effect'
        evoked_class = 'no effect'
        if p_val_baseline < 0.05:
            baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
        if p_val_evoked < 0.05:
            evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

        results.append({
            'Cluster': c,
            'Penetration': penetration,
            'Monkey Name': monkey_name,
            'Effect Size': AMI,
            'Layers': '',
            'Baseline Classification': baseline_class,
            'Evoked Classification': evoked_class,
            'Baseline p-value': p_val_baseline,
            'Evoked p-value': p_val_evoked,
            'Baseline High Firing Rate': baseline_high_firing_rate,
            'Baseline Low Firing Rate': baseline_low_firing_rate,
            'Evoked High Firing Rate': evoked_high_firing_rate,
            'Evoked Low Firing Rate': evoked_low_firing_rate
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
csv_filename = 'classification_results_2024-04-17.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
results_df.to_csv(csv_fullpath, index=False)


# Bar plot for population effect
mean_low_population = np.mean([res['Evoked Low Firing Rate'] for res in results])
mean_high_population = np.mean([res['Evoked High Firing Rate'] for res in results])
n_neurons = len([res['Evoked High Firing Rate'] for res in results])

# Standard errors
stderr_low_population = np.std([res['Evoked Low Firing Rate'] for res in results]) / np.sqrt(len(results))
stderr_high_population = np.std([res['Evoked High Firing Rate'] for res in results]) / np.sqrt(len(results))

plt.figure(figsize=(10, 6))

# Use matplotlib text properties instead of LaTeX formatting
labels = ['Low Arousal', 'High Arousal']
colors = ['black', 'red']
bars = plt.bar(labels, 
               [mean_low_population, mean_high_population], 
               yerr=[stderr_low_population, stderr_high_population], 
               color=colors,  
               capsize=5)

# Setting custom text properties for bar labels
bars[0].set_label(r'Low Arousal')
bars[1].set_label(r'High Arousal')

plt.title('p < 0.05')
plt.legend([f'Penetration Date: 2024-04-17'], loc='upper right', fontsize=12, frameon=False)
barplot_filename = 'population_effect.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()


# Scatter plot for high vs low 
plt.figure(figsize=(10, 6))
low_population_means = [res['Evoked Low Firing Rate'] for res in results]
high_population_means = [res['Evoked High Firing Rate'] for res in results]

# Determine the limit based on the maximum value across both axes
max_limit = max(max(low_population_means), max(high_population_means))
# Scatter plot for high vs low 
plt.figure(figsize=(10, 6))

# Scatter plot with low arousal in black and high arousal in red
plt.scatter(low_population_means, high_population_means, color='red')
plt.scatter(low_population_means, low_population_means, color='black')

# Determine the limit based on the maximum value across both axes
max_limit = max(max(low_population_means), max(high_population_means))
plt.plot([0, max_limit], [0, max_limit], 'k--')

plt.xlim(0, max_limit)
plt.ylim(0, max_limit)
plt.title('Pupil Diameter Effect')
plt.xlabel('Mean Firing Rate:Low')
plt.ylabel('Mean Firing Rate:High')
plt.legend([f'n = {n_neurons}', f'Penetration Date: 2024-04-17'], loc='upper left', fontsize=12, frameon=False)

scatterplot_filename = 'high_vs_low_arousal_Scatter_plot.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()



# Swarm plot 
plt.figure(figsize=(10, 6))
sns.swarmplot(x="Evoked Classification", y="Evoked High Firing Rate", data=results_df, color='red')
sns.swarmplot(x="Evoked Classification", y="Evoked Low Firing Rate", data=results_df, color='black')
plt.xticks(['up', 'down', 'no effect'])
plt.yscale('log')
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='High Arousal')
black_patch = plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Low Arousal')
plt.legend(handles=[red_patch, black_patch])
plt.title('Evoked Firing Rates')
swarmplot_filename = 'swarm_plot.svg'
swarmplot_fullpath = os.path.join(results_dir, swarmplot_filename)
plt.savefig(swarmplot_fullpath)
plt.show()















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import ttest_ind
from utils import utils
from lib import readSGLX
import os
import seaborn as sns

# results directory
results_dir = "C:/Users/mmustans/Documents/Projects/Neural_Response_Variability/results"

# Load data
spike_times_secpath = utils.getFilePath(windowTitle="Spike_times", filetypes=[('Spike times numpy file', '*.npy')])
clusters_path = utils.getFilePath(windowTitle="Select_clusters", filetypes=[('Clusters numpy file', '*.npy')])
stimulus_DF_path = utils.getFilePath(windowTitle="stimstart", filetypes=[('stimulus csv file', '*.csv')])
trial_DF_path = utils.getFilePath(windowTitle="trialstart", filetypes=[('trial csv file', '*.csv')])

spike_times_sec = np.load(spike_times_secpath)
clusters = np.load(clusters_path)
stimulusDF = pd.read_csv(stimulus_DF_path)
trialDF = pd.read_csv(trial_DF_path)

binFullPath = utils.getFilePath(windowTitle="Binary nidq file", filetypes=[("NIdq binary", "*.bin")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
pupilFullPath = utils.getFilePath(windowTitle="Pupil diameter data", filetypes=[("Numpy array", "*.npy")])

spike_times_clusters = utils.get_spike_times(clusters, spike_times_sec)

count_window = 20

def plot_raster(Y, ax, stimulusDF, title, pre_time, post_time, color):
    tr = 0
    for _, row in stimulusDF.iterrows():
        tr = tr + 1
        start_time = row['stimstart']        
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        ax.eventplot(spikes_in_trial - start_time, color=color, linewidths=0.5, lineoffsets=tr)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
  # Set the x-axis limits to the exact range you want
    ax.set_xlim(-pre_time, post_time)

def calculate_median_pupil_diameter(pupil_diameter, stimulusDF, baseline_time, sample_rate):
    medians = []
    for index, row in stimulusDF.iterrows():
        start_sample = int((row['stimstart'] - baseline_time) * sample_rate)
        stop_sample = int(row['stimstop'] * sample_rate)
        trial_data = pupil_diameter[start_sample:stop_sample]
        median_diameter = np.median(trial_data)
        medians.append(median_diameter)
    return np.array(medians)

def extract_spike_data_for_trials(Y, stimulusDF, pre_time=0.15, post_time=0.15, bin_size=0.001):
    trial_spike_data = []
    bin_edges = np.arange(-pre_time, post_time + bin_size, bin_size)
    expected_hist_length = len(bin_edges) - 1  # length hist array
    
    for i, row in stimulusDF.iterrows():
        start_time = row['stimstart']
        spikes_in_trial = Y[(Y >= start_time - pre_time) & (Y <= start_time + post_time)]
        
        if len(spikes_in_trial) == 0:
            # If no spikes append an array of zeros
            hist = np.zeros(expected_hist_length)
        else:
            #compute the hist
            spike_times_relative = spikes_in_trial - start_time
            hist, _ = np.histogram(spike_times_relative, bins=bin_edges)
        
        trial_spike_data.append(hist)
    
    trial_spike_data = np.array(trial_spike_data)
    
    return trial_spike_data

 
def meanvar_PSTH(data: np.ndarray, count_window: int = 100, style: str = 'same', 
                 return_bootdstrs: bool = False, nboots: int = 1000, binarize: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Binarize the data if binarize=True
    if binarize:
        data = data > 0
    
    if style == 'valid':
        mean_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        vari_timevector = np.nan * np.ones((data.shape[1] - count_window + 1))
        tmp = np.ones((data.shape[0], data.shape[1] - count_window + 1))
    else:
        mean_timevector = np.nan * np.ones((data.shape[1]))
        vari_timevector = np.nan * np.ones((data.shape[1]))
        tmp = np.ones((data.shape[0], data.shape[1]))
            
    for i in range(data.shape[0]):
        tmp[i,:] = np.convolve(data[i,:], np.ones(count_window,), style)
            
    vari_timevector = np.var(tmp, axis=0)
    mean_timevector = np.mean(tmp, axis=0)
    
    if return_bootdstrs:
        boot_inds = np.random.choice(tmp.shape[0], (tmp.shape[0], nboots))
        mean_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        vari_timevector_booted = np.nan * np.ones((nboots, tmp.shape[1]))
        for i in range(nboots):
            mean_timevector_booted[i,:] = np.mean(tmp[boot_inds[:,i],:], axis=0)
            vari_timevector_booted[i,:] = np.var(tmp[boot_inds[:,i],:], axis=0)
    else:
        mean_timevector_booted = np.array([])
        vari_timevector_booted = np.array([])
            
    return mean_timevector, vari_timevector, tmp, mean_timevector_booted, vari_timevector_booted

def get_spike_counts(spike_times, stim_times, pre_time, post_time, initial_time=0.035):
    baseline_counts = []
    evoked_counts = []
    
    for stim_time in stim_times:
        baseline_spikes = spike_times[(spike_times >= stim_time - pre_time) & (spike_times < stim_time)]
        evoked_spikes = spike_times[(spike_times >= stim_time + initial_time) & (spike_times < stim_time + post_time)]
       
        
        baseline_counts.append(len(baseline_spikes))
        evoked_counts.append(len(evoked_spikes))
    
    return np.array(baseline_counts), np.array(evoked_counts)

# Pupil diameter processing
pupil_diameter = np.load(pupilFullPath)
pupil_diameter = detrend(pupil_diameter)
pupil_diameter = pupil_diameter - np.nanmin(pupil_diameter)
pupil_diameter = pupil_diameter / np.nanmax(pupil_diameter)
pupil_trials = calculate_median_pupil_diameter(pupil_diameter, stimulusDF, 0.15, sRate)

pup_median = np.median(pupil_trials)
inds_low = np.where(pupil_trials < pup_median)[0]
inds_high = np.where(pupil_trials > pup_median)[0]

# Store spike data for AMI calculation
all_spike_data_low = []
all_spike_data_high = []

# Define baseline period 
pre_time  = 0.2
post_time = 0.2
initial_time=0.035

def calculate_baseline_rate(spike_times, baseline_pre_time):
    baseline_spikes = spike_times[spike_times < baseline_pre_time]
    baseline_rate = len(baseline_spikes) / baseline_pre_time
    return baseline_rate

results = []

# Define constants for penetration and monkey name
penetration = '2024-04-17'
monkey_name = 'Sansa'

for c in spike_times_clusters.keys():
    Y = spike_times_clusters[c]
    

    # Calculate baseline spike rate
    baseline_rate_count, evoked_rate_count = get_spike_counts(Y, stimulusDF["stimstart"].values, pre_time, post_time, initial_time)
    baseline_rate_mean= np.mean(baseline_rate_count)/pre_time
    evoked_rate_mean = np.mean(evoked_rate_count)/post_time

    if evoked_rate_mean >=  baseline_rate_mean + 5:
              
        # Create a single figure with subplots for rasters, PSTH, and Fano factor
        fig, axs = plt.subplots(3, 1, figsize=(10, 16))
        
        # Plot Rasters 
        plot_raster(Y, 
                    axs[0], 
                    stimulusDF.iloc[inds_low], 
                    title=f'Cluster {c} - Low Pupil Diameter',
                    pre_time=0.2,                    
                    post_time=0.2,
                    color='black')
        
        plot_raster(Y, 
                    axs[1], 
                    stimulusDF.iloc[inds_high], 
                    title=f'Cluster {c} - High Pupil Diameter',
                    pre_time=0.2,
                    post_time=0.2,
                    color='red')
        
        # Extract spike data for all trials
        spike_data_all = extract_spike_data_for_trials(Y, stimulusDF,pre_time,post_time)      

        # Ensure the indices 
        valid_inds_low = [i for i in inds_low if i < len(spike_data_all)]
        valid_inds_high = [i for i in inds_high if i < len(spike_data_all)]

        # Calculate PSTH for low and high pupil diameters 
        spike_data_low = spike_data_all[valid_inds_low]
        spike_data_high = spike_data_all[valid_inds_high]
        
        mean_psth_low, var_psth_low, _, _, _ = meanvar_PSTH(spike_data_low, count_window)
        mean_psth_high, var_psth_high, _, _, _ = meanvar_PSTH(spike_data_high, count_window)
        # Calculate total spike counts for each trial in low and high pupil states
        total_spike_counts_low = np.sum(spike_data_low, axis=1)
        total_spike_counts_high = np.sum(spike_data_high, axis=1)
        
        # Calculate mean firing rates for each condition
        baseline_low_firing_rate = np.mean(total_spike_counts_low)
        baseline_high_firing_rate = np.mean(total_spike_counts_high)

        # Calculate Fano factor
        fano_factor_low = var_psth_low / mean_psth_low
        fano_factor_high = var_psth_high / mean_psth_high

        # Ensure time_vector matches PSTH data length
        bin_size = 0.001  # 1 ms bins
        time_vector = np.arange(-pre_time, post_time, bin_size)        

        # Calculate standard error
        stderr_psth_low = np.sqrt(var_psth_low) / np.sqrt(len(valid_inds_low))
        stderr_psth_high = np.sqrt(var_psth_high) / np.sqrt(len(valid_inds_high))

        normalizing_factor = count_window/1000.0

        # Plot PSTH and Fano factor for low and high pupil diameters
        axs[2].plot(time_vector, mean_psth_low/normalizing_factor, color='black', linestyle='--', label='Low Pupil Diameter')
        axs[2].plot(time_vector, mean_psth_high/normalizing_factor, color='red', label='High Pupil Diameter')

        # Add fill_between for PSTH low
        axs[2].fill_between(time_vector, np.array(mean_psth_low - stderr_psth_low)/normalizing_factor, np.array(mean_psth_low + stderr_psth_low)/normalizing_factor, color='grey', alpha=0.3)
        # Add fill_between for PSTH high
        axs[2].fill_between(time_vector, np.array(mean_psth_high - stderr_psth_high)/normalizing_factor, np.array(mean_psth_high + stderr_psth_high)/normalizing_factor, color='lightcoral', alpha=0.3)

        axs[2].set_xlim(-pre_time, post_time)  
        axs[2].set_title(f'PSTH for Cluster {c}')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Spike Rate (Hz)')
        axs[2].legend()

        #axs[3].plot(time_vector, fano_factor_low, color='black', linestyle='--', label='Low Pupil Diameter')
        #axs[3].plot(time_vector, fano_factor_high, color='red', label='High Pupil Diameter')
        #axs[3].set_xlim(-pre_time, post_time) 
        #axs[3].set_title(f'Fano Factor for Cluster {c}')
        #axs[3].set_xlabel('Time (s)')
        #axs[3].set_ylabel('Fano Factor')
        #axs[3].legend()
        
        plt.tight_layout()
        svg_filename = f'cluster_{c}.svg'
        svg_fullpath = os.path.join(results_dir, svg_filename)
        plt.savefig(svg_fullpath)
        plt.close(fig)
        

        # Calculate spike counts for low and high pupil states
        baseline_counts_low, evoked_counts_low = get_spike_counts(Y, stimulusDF.iloc[inds_low]['stimstart'], pre_time, post_time)
        baseline_counts_high, evoked_counts_high = get_spike_counts(Y, stimulusDF.iloc[inds_high]['stimstart'], pre_time, post_time)
        
        # Calculate mean firing rates for each condition
        baseline_high_firing_rate = np.mean(baseline_counts_high)/pre_time
        baseline_low_firing_rate = np.mean(baseline_counts_low)/pre_time
        evoked_high_firing_rate = np.mean(evoked_counts_high)/post_time
        evoked_low_firing_rate = np.mean(evoked_counts_low)/post_time

        # Calculate AMI
        AMI = (evoked_high_firing_rate / evoked_low_firing_rate) if evoked_low_firing_rate != 0 else np.nan

        # Perform t-tests for baseline and evoked counts
        t_stat_baseline, p_val_baseline = ttest_ind(baseline_counts_low, baseline_counts_high, equal_var=False)
        t_stat_evoked, p_val_evoked = ttest_ind(evoked_counts_low, evoked_counts_high, equal_var=False)

        # Classify based on p-values
        baseline_class = 'no effect'
        evoked_class = 'no effect'
        if p_val_baseline < 0.05:
            baseline_class = 'up' if baseline_high_firing_rate > baseline_low_firing_rate else 'down'
        if p_val_evoked < 0.05:
            evoked_class = 'up' if evoked_high_firing_rate > evoked_low_firing_rate else 'down'

        # Append results to the list
        results.append({
            'Cluster': c,
            'Penetration': penetration,
            'Monkey Name': monkey_name,
            'Effect Size': AMI,  # Store the AMI here
            'Layers': '',
            'Baseline Classification': baseline_class,
            'Evoked Classification': evoked_class,
            'Baseline p-value': p_val_baseline,
            'Evoked p-value': p_val_evoked,
            'Baseline High Firing Rate': baseline_high_firing_rate,
            'Baseline Low Firing Rate': baseline_low_firing_rate,
            'Evoked High Firing Rate': evoked_high_firing_rate,
            'Evoked Low Firing Rate': evoked_low_firing_rate
        })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
csv_filename = 'classification_results_2024-04-17.csv'
csv_fullpath = os.path.join(results_dir, csv_filename)
# Save the DataFrame as a CSV file in the results directory
results_df.to_csv(csv_fullpath, index=False)


# Bar plot for population effect
mean_low_population = np.mean([res['Evoked Low Firing Rate'] for res in results])
mean_high_population = np.mean([res['Evoked High Firing Rate'] for res in results])



# Standard errors
stderr_low_population = np.std([res['Evoked Low Firing Rate'] for res in results]) / np.sqrt(len(results))
stderr_high_population = np.std([res['Evoked High Firing Rate'] for res in results]) / np.sqrt(len(results))

plt.figure(figsize=(10, 6))
plt.bar([r'$\mathbf{\textcolor{black}{Low\ Arousal}}$', r'$\mathbf{\textcolor{red}{High\ Arousal}}$'], 
        [mean_low_population, mean_high_population], 
        yerr=[stderr_low_population, stderr_high_population], 
        color=['black', 'red'],  
        capsize=5)
plt.title('p < 0.05')
plt.legend([f'Penetration Date: 2024-04-17'], loc='upper right', fontsize=12)
barplot_filename = 'population_effect.svg'
barplot_fullpath = os.path.join(results_dir, barplot_filename)
plt.savefig(barplot_fullpath)
plt.show()


# Scatter plot for high vs low 
plt.figure(figsize=(10, 6))
low_population_means = [res['Evoked Low Firing Rate'] for res in results]
high_population_means = [res['Evoked High Firing Rate'] for res in results]
n_neurons = len(low_population_means)
# Determine the limit based on the maximum value across both axes
max_limit = max(max(low_population_means), max(high_population_means))
plt.scatter(low_population_means, high_population_means, color='red')  
plt.scatter(low_population_means, low_population_means, color='black')
plt.plot([0, max_limit], [0, max_limit], 'k--')
plt.xlim(0, max_limit)
plt.ylim(0, max_limit)
plt.title('Pupil Diameter Effect')
plt.xlabel(r'Mean Firing Rate: $\mathbf{\textcolor{black}{Low}}$')
plt.ylabel(r'Mean Firing Rate: $\mathbf{\textcolor{red}{High}}$')
plt.legend([f'n = {n_neurons}', f'Penetration Date: 2024-04-17'], loc='upper left', fontsize=12)
scatterplot_filename = 'high_vs_low_arousal_Scatter_plot.svg'
scatterplot_fullpath = os.path.join(results_dir, scatterplot_filename)
plt.savefig(scatterplot_fullpath)
plt.show()


# Swarm plot 
plt.figure(figsize=(10, 6))
sns.swarmplot(x="Evoked Classification", y="Evoked High Firing Rate", data=results_df, color='red')
sns.swarmplot(x="Evoked Classification", y="Evoked Low Firing Rate", data=results_df, color='blue')
plt.xticks(['up', 'down', 'no effect'])
plt.yscale('log')
red_patch = plt.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='High Arousal')
black_patch = plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Low Arousal')
plt.legend(handles=[red_patch, blue_patch])
plt.title('Evoked Firing Rates')
plt.xlabel('Evoked Classification')
plt.ylabel('Evoked Firing Rate (Hz)')
swarmplot_filename = 'swarm_plot.svg'
swarmplot_fullpath = os.path.join(results_dir, swarmplot_filename)
plt.savefig(swarmplot_fullpath)
plt.show()





