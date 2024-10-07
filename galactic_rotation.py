"""
Radio Practical to determine maximal rotational velocities at 3 different
galactic longitudes in the Milky Way
Author: Nathan Ngqwebo
Version 1.0, October 5th 2024
Built in Python 3.12 (older versions may complain about type hints like: `list[list[dict]]`)

Uses:
HI profiles from University of Bonn https://www.astro.uni-bonn.de/hisurvey/euhou/LABprofile/index.php
"""

import matplotlib.pyplot as plt
import helpers as hp
import re
import os

# paths to observation and reference data
data_dir = os.getcwd() + '\\Observation\\FITS'
ref_dir = os.getcwd() + '\\l_297_b_-15'
bonn_dir = os.getcwd() + '\\bonn_data'

# extracting filenames
obs_file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.fits')] # observation filenames
grouped_obs_file_names = [obs_file_names[i:i+3] for i in range(0, len(obs_file_names), 3)]
ref_file_names = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.fits')] # reference filenames
bonn_file_names = [os.path.join(bonn_dir, f) for f in os.listdir(bonn_dir) if f.endswith('.txt')] # bonn data filenames

# read FITS tables and put them into dictionaries
grouped_obs_data_dicts = [[hp.read_fits_table(f) for f in group] for group in grouped_obs_file_names] # Use nested list comprehension
ref_data = [hp.read_fits_table(f) for f in ref_file_names]

# stack data over multiple FITS files with same target
stacked_obs = hp.stack_obs_by_longitude(grouped_obs_data_dicts)
stacked_ref = hp.stack_ref_by_longitude(ref_data)
# print(stacked_ref)
# reading bonn data for our longitudes
longitudes = hp.extract_gal_longitudes([d for sublist in grouped_obs_data_dicts for d in sublist])
HI_profiles = {}
for file in bonn_file_names:
    reg = re.search(r'(gal_l\d+|ref\d*)', file).group()
    print(reg)
    HI_profiles[f"{reg}"] = hp.read_Bonn_data(file)

# apply frequency window around 21 cm line
hp.window_obs(stacked_obs, HI_profiles['ref']['frequency'].min(), HI_profiles['ref']['frequency'].max())
hp.window_ref(stacked_ref, HI_profiles['ref']['frequency'].min(), HI_profiles['ref']['frequency'].max())

# normalising and flooring counts around 21 cm line
hp.normalise_floor_obs(stacked_obs)
hp.normalise_floor_ref(stacked_ref)

# # brightness temperature scaling
hp.bt_scale_obs(stacked_obs, HI_profiles)
bt_scale_factor = hp.get_bt_scale_factor(stacked_ref['RIGHT_POL'], HI_profiles['ref'])
hp.bt_scale_ref(stacked_ref, bt_scale_factor)

# correct each observation and the reference spectrum for LSR motion
HI_rest = 1420.40575177e6 # rest frequency of 21 cm line
vlsr = hp.get_LSR_correction(ref_data[0]['HEADER'])
hp.correct_obs_vlsr(stacked_obs, HI_rest)
hp.correct_ref_vlsr(stacked_ref, vlsr.si.value, HI_rest)

# plots
fig, ax = plt.subplots(1)
for i, longitude in enumerate(stacked_obs.values()):
    if i != 1:
        continue
    ax.plot(longitude['LEFT_POL']['frequency'], longitude['LEFT_POL']['counts'], label=f'{longitude['GAL_LONG']}')
    ax.plot(HI_profiles[longitude['GAL_LONG']]['frequency'], HI_profiles[longitude['GAL_LONG']]['BT'], label=f'Bonn HI Profile {longitude['GAL_LONG']}', linestyle='--')

ax.vlines(x=HI_rest, ymin=0, ymax=ax.get_ylim()[1], colors='k', alpha=0.7, linestyles='--', label='21cm rest frequency')
ax.plot()
ax.legend()
plt.show(block=True)

