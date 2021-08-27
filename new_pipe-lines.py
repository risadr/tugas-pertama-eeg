#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:27:45 2021

@author: new
"""
#import package
import os
from os.path import join as opj
import pathlib 
import numpy as np
import scipy
from scipy import signal 
import mne 
from os.path import join 
from scipy import io 
# from sys import path
# from pathlib import Path
# from os.path import dirname as dir
# participant = Path()
# print(participant)
from statistics import stdev 
from addingg_event_trial_03_17 import analyseBehaviourParticipant
#from addingg_event_trial import analyseEEGParticipant
from function_library_labelling_tr import (acc_check,pure_cond,tr_2_tr_cond,detect_outlier,detect_missing_trial)
import pandas as pd
from mne.preprocessing import(ICA, create_eog_epochs, corrmap)
%matplotlib qt5
import matplotlib.pyplot as plt

#which participant
sub= 'sub-00'
#read raw files

#file EEG
data_dir= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/RAW/data_exp_eeg/'

vhdr_dir = os.path.join(data_dir, 'S05_5.vhdr') 
vmrk_dir = os.path.join(data_dir, 'S05_5.vmrk') 
eeg_dir = os.path.join(data_dir, 'S05_5.eeg') 
#read raw 
raw = mne.io.read_raw_brainvision(vhdr_dir,eog=['LEOG','REOG','EMG1'], preload=True)
#check the layout cap
layout_from_raw = mne.channels.make_eeg_layout(raw.info)
layout_from_raw.plot()
events_from_annot, event_dict = mne.events_from_annotations(raw)
# Set  montage
montage = mne.channels.make_standard_montage('standard_1020')
#plot raw and define bads according to impadance vhdr
raw.plot()
raw_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/raw/'
raw.save(raw_path + sub +'_color-switching-raw.fif', overwrite=True) 

#define bads🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
raw_bads= raw.copy()
raw_bads.info['bads']= ['Fp1', 'O2', 'F7', 'F8', 'P8', 'Pz', 'FC1', 'CP1', 
                'CP2','FC5', 'CP5', 'CP6', 'POz', 'F1', 'F2', 'C2', 'PO3', 'PO4', 
                'CP4', 'F5', 'F6','P5', 'AF7', 'AF8', 'FT7', 'FT8', 
                'TP7', 'TP8', 'PO7', 'Fpz', 'AF3']
fig = raw_bads.plot()
#plt.close(fig)
raw_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/define_bads/'
raw_bads.save(raw_path+sub+'_define_bads_color-switching-raw.fif',overwrite=True)

#apply detrend🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
raw_detrend= raw_bads.copy()
x = raw_detrend.get_data()
   #detrending the signal
mn = np.mean(x, axis=-1, keepdims=True)
x = signal.detrend(x, axis=- 1, type='linear', bp=0, overwrite_data=False)
x = x+mn
#adding data back to raw object
raw_detrend._data = x
#plot
y = np.arange(1, len(raw_bads.get_data()[35])+1)
plt.plot(y, raw_bads.get_data()[35])
plt.plot(y, raw_detrend.get_data()[35])

#save
detrend_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/detrend/'
raw_detrend.save( detrend_path + sub +'_color-switching_detrended-raw.fif',overwrite=True)
       

#filter🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
raw_filter =raw_detrend.copy()
raw_filter.filter(0.05, 30., fir_design='firwin')
raw_filter.plot_psd(fmax=50)

#save
fil_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/filter/'
raw_filter.save(fil_path + sub + '_color-switching_filtered-raw.fif',overwrite=True)
       
#change trigger🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
def change_trigger(sub):
        
    print('Processing',sub)
    #setting the path for the required(rereferenced) files
    #mat_path = '/Users/new/Documents/new_office_folder/Documents/Bids_format_EEG/RAW/data_exp_eeg_bv/'
    
    mat_path= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/rawdata/'+sub+'/behavioral_data/'
    eeg_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/Filter/'
    mat_file= opj(mat_path,sub+'_color_task_manual.mat')
    print(mat_file)
    eeg_file= opj(eeg_path,sub+'_color-switching_filtered-raw.fif')
    print(eeg_file)
    #open script adding_event-trial
    analyze_behaviour = analyseBehaviourParticipant(mat_file)   
    #read eeg file
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    #read events
    events, event_ids = mne.events_from_annotations(raw)
    # making condition of trial type of eeg file
    
    trial_list_eeg = events       
    pure_trial = trial_list_eeg[:90]
    for index,item in enumerate(trial_list_eeg[:90]):
        print("Line {},  Value : {}".format(index, item))
        
    mix_trial = trial_list_eeg[90:]
    for index,item in enumerate(trial_list_eeg[90:]):
        print("Line {},  Value : {}".format(index, item))
    #creating the data frame    
    df0= pd.DataFrame(pure_trial, columns=['events', 'zero', 'conditions'])
    df =pd.DataFrame(mix_trial, columns=['events', 'zero', 'conditions'])
    #combining frame 
    trial_list_eeg = pd.concat([df0,df], ignore_index=True)    
    trial_list = analyze_behaviour['trial_list']     
    # change the trigger
    
    ori_and_modif_conditions = trial_list_eeg.join(trial_list["label_number_cond"])
    modified_conditions =    ori_and_modif_conditions.drop('conditions', 1)
    modified_mix_trial = modified_conditions.iloc[90:].to_numpy()
    
    new_event_idi = dict({'stay_pp':3, 'stay_ss':4, 'switch_ps':5, 'switch_sp':6,'ft':999,'Ict':90})
        
    fig = mne.viz.plot_events(modified_mix_trial,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_id=new_event_idi)
    
    #additional
    modified_all_trials = modified_conditions.iloc[:].to_numpy()
    
    new_event_idbi = dict({'pure_pref':1,'pure_sim':2,'stay_pp':3, 'stay_ss':4, 'switch_ps':5, 'switch_sp':6,'ft':999,'Ict':90})   
     
    fig = mne.viz.plot_events(modified_all_trials,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_id=new_event_idbi)

    #modified_all_trials = {'modified_all_trials':modified_all_trials, 'raw':raw}

    #converting to raw via annotations_from_events
    
    mapping = dict({1: 'pure_pref', 2: 'pure_sim', 3: 'stay_pp',
           4: 'stay_ss', 5: 'switch_ps', 6: 'switch_sp',9:"outlier",99:'missing',999:"ft",90:"Ict"})
    
    annot_from_events = mne.annotations_from_events(modified_all_trials,first_samp=raw.first_samp,
                              sfreq=raw.info['sfreq'],event_desc=mapping)
    #create parameter to change raw    
    modified_all_trials=  modified_all_trials[~np.isnan(modified_all_trials).any(axis=1)]
    new_eventid = np.unique(modified_all_trials[:,2])
    onsets = modified_all_trials[:,0] /raw.info['sfreq']
    durations = np.zeros_like(onsets)  
    descriptions = [mapping[new_eventid] for new_eventid in modified_all_trials[:,2]]
    #applying given parameter
    annot_from_events = mne.Annotations(onset= onsets, duration= durations, description=descriptions, orig_time = raw.info['meas_date']) 
    new_raw= raw.copy()
    #set the change
    new_raw.set_annotations(annot_from_events)
    #checking the changes
    events, event_ids = mne.events_from_annotations(new_raw)
    
    #setting the event_ids via events_from_annotations
    events, _= mne.events_from_annotations(new_raw, event_id={'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
            'stay_ss':4, 'switch_ps':5 ,'switch_sp':6,'ft':999,'Ict':90})
    #event dict epoching
    event_dict = {'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
                'stay_ss':4, 'switch_ps':5 ,'switch_sp':6,"ft":999,"Ict":90}
    # picks_eeg = mne.pick_types(new_raw.info, meg=False, eeg=True, eog=True,
    #                        stim=False, exclude='bads')

    epochs= mne.Epochs(new_raw, events, tmin=-.1, tmax=1.,
            event_id={'pure_pref': 1,'pure_sim':2,'stay_pp': 3,
                      'stay_ss':4, 'switch_ps':5 ,'switch_sp':6} ,
                    preload=True, baseline=(-0.1, 0))
    #epochs.plot()
    
    change_trigger_path= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/change_trigger/'
    epochs.save(change_trigger_path+sub+'_epochs_task-switching-epo.fif',overwrite=True)

'''       
#run for all subjects
for sub in subjects:
    change_trigger(sub) 
'''    
#remove ICA 🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
change_trigger_file= opj(change_trigger_path,sub+'_epochs_task-switching-epo.fif')
epochs = mne.read_epochs(change_trigger_file, preload=True)
epochs.plot()
ica=ICA(n_components=25,method="fastica",random_state= 1,max_iter=200).fit(epochs)
a= ica.plot_sources(epochs)
b = ica.plot_components(inst=epochs)
#exclude
ica.exclude = [0,1,19] 
epochs_ica = epochs.copy()
ica.apply(epochs_ica)

epochs_ica.plot()

ICA_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/ICA_analysis/'
epochs_ica.save(ICA_path+sub+'_color-switching_fit-epo.fif',overwrite=True)

#interpolate🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
epochs_interpolate= epochs_ica.copy()
#filter the raw
epochs_interpolate.interpolate_bads(reset_bads=True, mode='accurate')
#plot
epochs_interpolate.plot()
int_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/interpolated_channels/'
epochs_interpolate.save(int_path + sub + '_color-switching_interpolate-epo.fif',overwrite=True)
  
#Artifact Reject🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
new_reject=dict(eeg=150e-6)
clean_epochs= epochs_interpolate.copy()
clean_epochs = clean_epochs.drop_bad(reject=new_reject)
   
clean_epochs.plot_drop_log() 
    
artifact_rejection_path = '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/remove_artifact/'
clean_epochs.save(artifact_rejection_path + sub +'_Artifact_rejected-epo.fif',overwrite=True)
 
#checking evoked  ERP🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
df= pd.read_csv(os.path.join('/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/participants.tsv'),\
                                 index_col=0,delimiter='\t')
subjects=df.index.values
n_subs=len(subjects)

bs=[]
#for sub in subjects:
#    print('Processing',sub)
change_trigger_path= '/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/remove_artifact/'    #epochs_file = opj(epochs_path,sub+'_color-switching_epochs_equalized-epo.fif')
epochs_file = opj(change_trigger_path,sub + '_Artifact_rejected-epo.fif')
bs.append(mne.read_epochs(epochs_file, preload=True))
    
#average and plot
ab= bs[0]['stay_pp'].average()
bb= bs[0]['switch_sp'].average()
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks='eeg', axes='topo')

#path🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽🔽
fig_path='/Volumes/HD710 PRO/work_folder/Bids_format_EEG/switching_task/new_pipeline/preprocessing/'+sub+'/ERP/'
#plot epochs
fig= bs[0].plot()
plt.savefig(fig_path + "epochs.png", dpl =600)

#Fpz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['Fpz'])
plt.savefig(fig_path + "evoked_preference_Fpz.png", dpl =600)

#Fz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['Fz'])
plt.savefig(fig_path + "evoked_preference_Fz.png", dpl =600)

#Cz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['Cz'])
plt.savefig(fig_path + "evoked_preference_Cz.png", dpl =600)

#CPz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['CPz'])
plt.savefig(fig_path + "evoked_preference_CPz.png", dpl =600)

#Pz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['Pz'])
plt.savefig(fig_path + "evoked_preference_Pz.png", dpl =600)

#POz
fig= mne.viz.plot_compare_evokeds(dict(stay_pp= ab , switch_sp= bb), picks=['POz'])
plt.savefig(fig_path + "evoked_preference_POz.png", dpl =600)


