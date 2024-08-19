from lib import readSGLX
from utils import utils
import numpy as np
import subprocess 
import pandas as pd
from time import sleep
import os

CatGT_dir  = 'C:/Users/mmustans/Documents/GitHub/CatGTWinApp/CatGT-win/'
TPrime_dir = 'C:/Users/mmustans/Documents/GitHub/TPrimeWinApp/TPrime-win/'

# catGT parameters
DIR = ' -dir=D:/Data/MM001_Sansa/Electrophysiology/2024-04-18'
RUN = ' -run=BSD'
prs = ' -g=0 -t=0 -ap -ni -prb_fld -prb=0'
xa3 = ' -xa=0,0,3,2.5,1,0'  # trial start
xia = ' -xia=0,0,3,2.5,3.5,0'  # trial stop
xa = ' -xa=0,0,5,2.5,1,0'  # stim start
xia3 = ' -xia=0,0,5,2.5,3,0'  # stim stop


# run CatGT
os.chdir('C:/Users/mmustans/Documents/GitHub/CatGTWinApp/CatGT-win/')
print('CatGT is running, please wait for the process to finish')

os.chdir('C:/Users/mmustans/Documents/GitHub/TPrimeWinApp/TPrime-win/')

subprocess.run('CatGT' + DIR + RUN + prs + xia + xa +xa3+xia3, shell=True, capture_output=True)

# convert spike times to seconds
binFullPath = utils.getFilePath(windowTitle="Select binary ap file", filetypes=[("sGLX binary", "*.bin")])
spikesFullPath = utils.getFilePath(windowTitle="Select spikes file", filetypes=[("KS output spikes_times", "*.npy")])
meta = readSGLX.readMeta(binFullPath)
sRate = readSGLX.SampRate(meta)
spike_times_smp = np.load(spikesFullPath)
spike_times_sec = np.around(np.divide(spike_times_smp, sRate, dtype=float), decimals=6)
np.save(spikesFullPath.with_stem('spike_times_sec'), spike_times_sec)

# TPrime parameters
tostream = utils.getFilePath(windowTitle="SYNC tostream file (IMEC0 edgefile.txt)", filetypes=[("CatGT output", "*.txt")])
fromstream = utils.getFilePath(windowTitle="SYNC fromstream file (usually NIDAQ edgefile.txt)", filetypes=[("CatGT output", "*.txt")])
trialstart = utils.getFilePath(windowTitle="SYNC trial start (usually xa3)", filetypes=[("CatGT output", "*.txt")])
trialstop = utils.getFilePath(windowTitle="SYNC trial stop (usually xia3)", filetypes=[("CatGT output", "*.txt")])
stimstart = utils.getFilePath(windowTitle="SYNC stimulus start (usually xa5)", filetypes=[("CatGT output", "*.txt")])
stimstop = utils.getFilePath(windowTitle="SYNC stimulus stop (usually xa5)", filetypes=[("CatGT output", "*.txt")])


syncperiod = ' -syncperiod=1.0'
tostream = ' -tostream=' + str(tostream)
fromstream = ' -fromstream=1,' + str(fromstream)
trialstart = ' -events=1,' + str(trialstart) + ',' + str(trialstart)[0:len(str(trialstart)) - len(str(trialstart.stem) + '.txt')] + 'trialstart.txt'
trialstop = ' -events=1,' + str(trialstop) + ',' + str(trialstop)[0:len(str(trialstop)) - len(str(trialstop.stem) + '.txt')] + 'trialstop.txt'
stimstart = ' -events=1,' + str(stimstart) + ',' + str(stimstart)[0:len(str(stimstart)) - len( str(stimstart.stem) + '.txt')] + 'stimstart.txt'
stimstop = ' -events=1,' + str(stimstop) + ',' + str(stimstop)[0:len(str(stimstop)) - len(str(stimstop.stem) + '.txt')] + 'stimstop.txt'

# run TPrime 
subprocess.run('cd ' + TPrime_dir, shell=True)
subprocess.run('TPrime' + syncperiod + tostream + fromstream + trialstart + trialstop + stimstart + stimstop, shell=True)

# get paths to the pulse files 
trialstartFullPath = utils.getFilePath(windowTitle="Select trialstart file", filetypes=[("TPrime output", "*.txt")])
trialstopFullPath = utils.getFilePath(windowTitle="Select trialstop file", filetypes=[("TPrime output", "*.txt")])
stimstartFullPath = utils.getFilePath(windowTitle="Select stimstart file", filetypes=[("TPrime output", "*.txt")])
stimstopFullPath = utils.getFilePath(windowTitle="Select stimstop file", filetypes=[("TPrime output", "*.txt")])


# read the pulse files and convert to dataframe
trialstartDF = pd.read_csv(trialstartFullPath.absolute(), sep=" ", header=None)
trialstartDF.columns = ['trialstart']

trialstopDF = pd.read_csv(trialstopFullPath.absolute(), sep=" ", header=None)
trialstopDF.columns = ['trialstop']

trialsDF = trialstartDF.join(trialstopDF)

stimstartDF = pd.read_csv(stimstartFullPath.absolute(), sep=" ", header=None)
stimstartDF.columns = ['stimstart']

stimstopDF = pd.read_csv(stimstopFullPath.absolute(), sep=" ", header=None)
stimstopDF.columns = ['stimstop']

stimstopDF['stimstop'] = stimstartDF['stimstart'] + 0.5




# save the dataframes
trialsDF.to_csv(trialstartFullPath.with_suffix('.csv'), index=False)
stimstartDF.to_csv(stimstartFullPath.with_suffix('.csv'), index=False)
stimstopDF.to_csv(stimstopFullPath.with_suffix('.csv'), index=False)


