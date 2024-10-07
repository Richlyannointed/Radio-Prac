"""
Radio Practical to determine maximal rotational velocities at 3 different
galactic longitudes in the Milky Way
Author: Nathan Ngqwebo
Version 1.0, October 5th 2024
Built in Python 3.12 (older versions may complain about type hints like: `list[list[dict]]`)

Uses:
HI profiles from University of Bonn https://www.astro.uni-bonn.de/hisurvey/euhou/LABprofile/index.php
"""
from astropy.utils.exceptions import AstropyWarning
from astropy import constants as const
from correctLSR import vlsr_calc
from astropy.time import Time
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import numpy as np
import warnings
import re
import os
warnings.simplefilter('ignore', category=AstropyWarning)

# paths to observation and reference data
data_dir = os.getcwd() + '\\Observation\\FITS'
ref_dir = os.getcwd() + '\\l_297_b_-15'
bonn_dir = os.getcwd() + '\\bonn_data'

def read_Bonn_data(filepath:str)-> pd.DataFrame:
    data = np.loadtxt(filepath, skiprows=4)
    df = pd.DataFrame({'v_lsr': data[:,0],
                       'BT': data[:,1],
                       'frequency': data[:,2] * 1e6, # convert MHz to Hz
                       'wavelength': data[:, 3] * 1e-2, # convert cm to m
                       }).astype(float)
    return df


def process_pol(data, key:str, statuses) -> np.ndarray:
    """Helper function to process polarimetry data."""
    pol_data = data[key]
    pol_data = np.column_stack((statuses, pol_data))  # Add 'status' column
    pol_data = pol_data[np.char.strip(pol_data[:, 0]) != 'cal']  # Filter out calibrations
    return np.sum(pol_data[:, 1:].astype(float), axis=0)  # Sum measurements across the frequencies


def read_fits_table(filepath: str) -> dict:
    """Extracts dictionary of FITS data and stacks 'on', 'off' spectra into 1D array"""
    with fits.open(filepath) as hdul:
        # Extract FITS data into a dictionary by col.name : table
        data = {column.name: hdul[1].data[column.name] for column in hdul[1].columns}
        
        statuses = data['STATUS']  # Extract statuses

        # Process both RIGHT_POL and LEFT_POL
        data['RIGHT_POL'] = process_pol(data, 'RIGHT_POL', statuses)
        data['LEFT_POL'] = process_pol(data, 'LEFT_POL', statuses)
        data['HEADER'] = hdul[1].header
        return data
    

def extract_gal_longitudes(obs_data: list[dict]) -> list:
    longitudes = set()

    for i, table in enumerate(obs_data):
        try:
            """ DEBUGGING
            print(np.mean(table['Gal_Long']))
            print(np.mean(table['Gal_Lat']))
            print(table['HEADER']['BMAJ'])
            print()
            """

            longitudes.add(np.round(np.mean(table['Gal_Long']), 2)) # store all unique longitudes up to 2nd decimal

        except TypeError:  # Catch TypeError if the value is not roundable
            print('failed to round longitude')
            return None  # or handle the error more gracefully as needed
    return longitudes


def stack_ref_by_longitude(ref_data: list[dict]) -> dict :
    """Summs all right, and left_pol data, calculates average target longitude, and latidude
    
    params
    ------
    obs_data : list of lists of grouped observation dictionaries eg., 
                [ [{long_28_A},{long_28_B}, {long_28_C}, ...] ]
    """
    result = {}
    # Initialize arrays to zero based on the shape of the first observation's data
    summed_left_pol = np.zeros_like(ref_data[0]['LEFT_POL'])
    summed_right_pol = np.zeros_like(ref_data[0]['RIGHT_POL'])
    

    # extract unique galactic longitudes in our observations
    # Stack counts over multiple observations
    for data in ref_data:
        summed_left_pol += data['LEFT_POL']
        summed_right_pol += data['RIGHT_POL']
    
    header = ref_data[0]['HEADER']
    central_freq = 1428.75e6 # corrected central frequency in Hz
    bandwidth = header['BANDWID'] *1e6 # bandwidth in Hz
    # band_res = header['BNDRES'] # band resolution in Hz
    # freq = np.linspace(central_freq - bandwidth/2 + bandwidth/(2*1024), 
    #                    central_freq + bandwidth/2 - bandwidth/(2*1024), 
    #                    1024)  # create frequency axis
    freq = np.linspace(central_freq - bandwidth/2 , 
                           central_freq + bandwidth/2 , 
                           1024)  # create frequency axis
    left_pol = np.column_stack((freq, summed_left_pol)) # stack polimetry
    right_pol = np.column_stack((freq, summed_right_pol))
    
    result['LEFT_POL'] = pd.DataFrame({'frequency': left_pol[:,0], 'counts': left_pol[:,1]})
    result['RIGHT_POL'] = pd.DataFrame({'frequency': right_pol[:,0], 'counts': right_pol[:,1]})
    return result

def stack_obs_by_longitude(obs_data: list[list[dict]]) -> dict :
    """Summs all right, and left_pol data, calculates average target longitude, and latidude
    
    params
    ------
    obs_data : list of lists of grouped observation dictionaries eg., 
                [ [{long_28_A},{long_28_B}, {long_28_C}, ...], [{long_300_A}, long_300_B}, long_300_C}, ...], ... ]
    """
    result = {}
    for longitudes in obs_data:
        # Initialize arrays to zero based on the shape of the first observation's data
        summed_left_pol = np.zeros_like(longitudes[0]['LEFT_POL'])
        summed_right_pol = np.zeros_like(longitudes[0]['RIGHT_POL'])
        longitude_str = list(extract_gal_longitudes(longitudes))[0]

        # extract unique galactic longitudes in our observations
        # Stack counts over multiple observations
        for data in longitudes:
            summed_left_pol += data['LEFT_POL']
            summed_right_pol += data['RIGHT_POL']
        
        header = longitudes[0]['HEADER']
        central_freq = 1428.75e6 # corrected central frequency in Hz
        bandwidth = header['BANDWID'] # bandwidth in Hz
        band_res = header['BNDRES'] # band resolution in Hz
        # freq = np.linspace(central_freq - bandwidth/2 + band_res/2, 
        #                    central_freq + bandwidth/2 - band_res/2, 
        #                    1024)  # create frequency axis
        freq = np.linspace(central_freq - bandwidth/2 , 
                           central_freq + bandwidth/2 , 
                           1024)  # create frequency axis
        left_pol = np.column_stack((freq, summed_left_pol)) # stack polimetry
        right_pol = np.column_stack((freq, summed_right_pol))
        
        result[f'long{longitude_str:.0f}'] = {
            'GAL_LONG':longitude_str ,
            'LEFT_POL': pd.DataFrame({'frequency': left_pol[:,0], 'counts': left_pol[:,1]}),
            'RIGHT_POL': pd.DataFrame({'frequency': right_pol[:,0], 'counts': right_pol[:,1]})
            }
    return result


def window_obs(data: dict, f_min: float, f_max: float) -> None:
    """Window observation data around 21cm line based on frequency range of Bonn data"""
    for info in data.values():
        for pole, df in info.items():
            if pole == 'GAL_LONG':
                continue
            info[pole] = df[(df['frequency'] > f_min) & (df['frequency'] < f_max)]
        # df1 = info['LEFT_POL']
        # df2 = info['RIGHT_POL']
        # info['LEFT_POL'] = df1[(df1['frequency'] > f_min) & (df1['frequency'] < f_max)]
        # info['RIGHT_POL'] = df2[(df2['frequency'] > f_min) & (df2['frequency'] < f_max)]
    return


def window_ref(data: dict, f_min: float, f_max: float) -> None:
    """Window reference spectrum around 21cm line based on frequency range of Bonn data"""
    for pole, df in data.items():
        data[pole] = df[(df['frequency'] > f_min) & (df['frequency'] < f_max)]


def normalise_floor_obs(data:dict) -> None:
    """Normalise and floor observation data counts around 21cm line"""
    for longitude in data.values():
        for pole, df in longitude.items():
            if pole == 'GAL_LONG':
                continue
            longitude[pole]['counts'] = df['counts'] / df['counts'].sum() 
            longitude[pole]['counts'] -= df['counts'].mean()


def normalise_floor_ref(data:dict) -> None:
    """Normalise and floor reference data counts around 21cm line"""
    for pole, df in data.items():
        if pole == 'GAL_LONG':
            continue
        data[pole]['counts'] = df['counts'] / df['counts'].sum() 
        data[pole]['counts'] -= df['counts'].mean()


def get_bt_scale_factor(ref_pol:dict, HI_profile:pd.DataFrame) -> float:
    """Scale the correct polimetry data from the reference spectrum to the HI profile peak 
    to set Brightness Temperature scale"""
    f = HI_profile['BT'].max() / ref_pol['counts'].max()
    return f


def bt_scale_obs(data:dict, factor:float) -> None:
    """Apply brightness temp scale factor to observation data"""
    for longitude in data.values():
        for pole in longitude.keys():
            if pole == 'GAL_LONG':
                continue
            longitude[pole]['counts'] *= factor
    

def bt_scale_ref(data:dict, factor:float) -> None:
    """Apply brightness temp scale factor to reference data"""
    for pole in data.keys():
        if pole == 'GAL_LONG':
            continue
        data[pole]['counts'] *= factor


def get_LSR_correction(ref_header) -> float:
    """Wrapper for Wolfgang Herrmann, Astropeiler Stockert code which
    gets observer's velocity relative to Local Standard of Rest"""
    # print(ref_header['DATE-OBS'])
    # print(Time(ref_header['DATE-OBS'], scale='utc'))
    # Extracting the observation times from the FITS header
    date_obs = Time(ref_header['DATE-OBS'], scale='utc')
    date_end = Time(ref_header['DATE-END'], scale='utc')
    avg_time = date_obs + 0.5 * (date_end - date_obs)
    _, vlsr = vlsr_calc(obs_lon=ref_header['SITELONG'], 
                    obs_lat=ref_header['SITELAT'], 
                    obs_ht=ref_header['SITEELEV'], 
                    ra=ref_header['RA'],
                    dec=ref_header['DEC'], 
                    time=date_obs
                    )
    return vlsr 


def correct_obs_vlsr(data:dict, vlsr:float, f0:float) -> None:
    """Applies LSR relative velocity frequenc correction to observation data"""
    delta_f = (f0 * vlsr) / const.c.value
    for longitude in data.values():
        for pole in longitude.keys():
            if pole == 'GAL_LONG':
                continue
            longitude[pole]['frequency'] -= delta_f


def correct_ref_vlsr(data:dict, vlsr:float, f0:float) -> None:
    """Applies LSR relative velocity frequenc correction to reference data"""
    delta_f = f0 * (vlsr) / const.c.value
    for pole in data.keys():
        if pole == 'GAL_LONG':
            continue
        data[pole]['frequency'] -= delta_f


# extracting filenames
obs_file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.fits')] # observation filenames
grouped_obs_file_names = [obs_file_names[i:i+3] for i in range(0, len(obs_file_names), 3)]
ref_file_names = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.fits')] # reference filenames
bonn_file_names = [os.path.join(bonn_dir, f) for f in os.listdir(bonn_dir) if f.endswith('.txt')] # bonn data filenames

# read FITS tables and put them into dictionaries
grouped_obs_data_dicts = [[read_fits_table(f) for f in group] for group in grouped_obs_file_names] # Use nested list comprehension
ref_data = [read_fits_table(f) for f in ref_file_names]

# stack data over multiple FITS files with same target
stacked_obs = stack_obs_by_longitude(grouped_obs_data_dicts)
stacked_ref = stack_ref_by_longitude(ref_data)
# print(stacked_ref)
# reading bonn data for our longitudes
longitudes = extract_gal_longitudes([d for sublist in grouped_obs_data_dicts for d in sublist])
HI_profiles = {}
for file in bonn_file_names:
    HI_profiles[f"{re.search(r'(gal\d+|ref\d*)', file).group()}"] = read_Bonn_data(file)

# apply frequency window around 21 cm line
window_obs(stacked_obs, HI_profiles['ref']['frequency'].min(), HI_profiles['ref']['frequency'].max())
window_ref(stacked_ref, HI_profiles['ref']['frequency'].min(), HI_profiles['ref']['frequency'].max())

# normalising and flooring counts around 21 cm line
normalise_floor_obs(stacked_obs)
normalise_floor_ref(stacked_ref)

# # brightness temperature scaling
bt_scale_factor = get_bt_scale_factor(stacked_ref['RIGHT_POL'], HI_profiles['ref'])
bt_scale_obs(stacked_obs, bt_scale_factor)
bt_scale_ref(stacked_ref, bt_scale_factor)

# correct for LSR motion
HI_rest = 1420.40575177e6 # rest frequency of 21 cm line
vlsr = get_LSR_correction(ref_data[0]['HEADER'])
correct_obs_vlsr(stacked_obs, vlsr.si.value, HI_rest)
correct_ref_vlsr(stacked_ref, vlsr.si.value, HI_rest)

# plots
fig, ax = plt.subplots(1)
for longitude in stacked_obs.values():
    ax.plot(longitude['LEFT_POL']['frequency'], longitude['LEFT_POL']['counts'])

ax.plot(stacked_ref['RIGHT_POL']['frequency'], stacked_ref['RIGHT_POL']['counts'])
ax.plot(HI_profiles['ref']['frequency'], HI_profiles['ref']['BT'], label='Bonn HI Profile')
ax.plot()
# ax.hlines(y=0, color='r', linestyles='--', xmin=stacked_ref['RIGHT_POL']['frequency'].min(), xmax=stacked_ref['RIGHT_POL']['frequency'].max())
ax.legend()
plt.show(block=True)

# print(stacked_obs)
# print(stacked_obs['long28']['RIGHT_POL']['frequency'] > HI_profiles['ref']['frequency'].min())
# print(HI_profiles['ref']['frequency'].max())

