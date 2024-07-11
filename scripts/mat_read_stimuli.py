from utils import utils
import scipy.io as sio
import pandas as pd
import numpy as np

# Load MAT file using file dialog
matFilePath = utils.getFilePath(windowTitle="Select MAT file with trial data", filetypes=[("MAT files", "*.mat")])
mat = sio.loadmat(matFilePath, struct_as_record=False, squeeze_me=True)

# Extract trial records
expt_info = mat['expt_info']
trial_records = expt_info.trial_records

# Display meta parameters recorded for the penetration
meta_params = dir(expt_info)


# Display the reward scaler
reward_scaler = expt_info.reward_scaler


# Select stimulus CSV file using file dialog
stimulus_DF_path = utils.getFilePath(windowTitle="Select stimulus CSV file", filetypes=[('CSV files', '*.csv')])
stimulusDF = pd.read_csv(stimulus_DF_path)

# Extract and print trImage values for all trials
trImage_values = []
counter = 0
nan_count = 0
array_count = 0
float_count = 0
int_count = 0

for i, trial in enumerate(trial_records):
    trImage_value = trial.trImage if hasattr(trial, 'trImage') else None
    if isinstance(trImage_value, np.ndarray):
        counter += trImage_value.shape[0]
        array_count += trImage_value.shape[0]
    elif isinstance(trImage_value, float):
        if pd.isna(trImage_value):
            nan_count += 1
        else:
            counter += 1
            float_count += 1
    elif isinstance(trImage_value, int):
        counter += 1
        int_count += 1
    elif trImage_value is None:
        nan_count += 1
    trImage_values.append(trImage_value)
    

# Get the number of trImage and stimstart values
num_trImage_values = counter
num_stimstart_values = len(stimulusDF['stimstart'])

# Compare the numbers and print detailed counts
print(f"Number of trImage values: {num_trImage_values}")
print(f"Number of stimstart values: {num_stimstart_values}")
print(f"Counts - Arrays: {array_count}, Floats: {float_count}, Integers: {int_count}, NaNs: {nan_count}")

if num_trImage_values == num_stimstart_values:
    print("The number of trImage values matches the number of stimstart values.")
else:
    print("The number of trImage values does NOT match the number of stimstart values.")


















