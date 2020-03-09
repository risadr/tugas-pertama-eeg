# tugas-pertama-eeg
"""Take home assigment
by me March 9 2020"""
# open up the subject's vhdr file
f=open("/Users/macbook/Desktop/cc_ptsd_practice/sub-010002.vhdr","r")
#making variable for the file
lines=f.readlines()
#counting the number of the line
len(lines)
#Define the impedance
impedance=[]
#Looping the impedance line
for i in range(234-64, 234):
    impedance.append(lines[i])
#calling the impedance channel
impedance
#calling information from the raw file
import mne
mne.io
raw = mne.io.read_raw_brainvision("/Users/macbook/Desktop/cc_ptsd_practice/sub-010002.vhdr", preload=True)
#Show the raw info
raw.info
#selecting the channel names
channelnames=raw.info["ch_names"]
#calling the channel channel names
channelnames
#counting the channel channel names
len(channelnames)
#adding the missing list
channelnames.append("Gnd")
channelnames.append("Ref")
#checking the channel ch_names
channelnames
#selecting the value
valueofimpedance=[]
#Looping for value
for i in range(len(impedance)):
        valueofimpedance.append(int(impedance[i][-2]))
#Counting the value of impedance
len(valueofimpedance)
#making the dictionary for key and value
d_impedance=dict(zip(channelnames, valueofimpedance))
#checking the impedance
d_impedance
#defining variable of selected channel
higherthan1=[]
for key, value in d_impedance.items():
    if value >=1:
          higherthan1.append(key)
#checking the channels
higherthan1

