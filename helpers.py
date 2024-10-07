"""
helpers.py

This module contains helper functions for astronomical data processing and analysis.
"""

# Necessary imports
from astropy.utils.exceptions import AstropyWarning
from astropy import constants as const
from correctLSR import vlsr_calc
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)

if __name__ != '__main__':

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
                gal_long = np.round(np.mean(table['Gal_Long']), 0) # mean longitude up to whole number
                if gal_long > 0:
                    longitudes.add(gal_long)
                else:
                    longitudes.add(360 + gal_long)

            except TypeError:  # Catch TypeError if the value is not roundable
                print('failed to round longitude')
                return None  # or handle the error more gracefully as needed
        return longitudes


    def stack_obs_by_longitude(obs_data: list[list[dict]]) -> dict :
        """Summs all right, and left_pol data, calculates average target longitude, and latidude
        
        params
        ------
        obs_data : list of lists of grouped observation dictionaries eg., 
                    [ [{long_28_A},{long_28_B}, {long_28_C}, ...], [{long_300_A}, long_300_B}, long_300_C}, ...], ... ]
        
        returns
        -------
        dict : dictionary with structure: { 'GAL_LONG' : str (eg, 'gal_l##'),
                                            'LEFT_POL': DataFrame
                                            'RIGHT_POL': DataFrame
                                            }
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
            
            result[f'gal{longitude_str:.0f}'] = {
                'HEADER' : header,
                'GAL_LONG': f'gal_l{longitude_str:.0f}' ,
                'LEFT_POL': pd.DataFrame({'frequency': left_pol[:,0], 'counts': left_pol[:,1]}),
                'RIGHT_POL': pd.DataFrame({'frequency': right_pol[:,0], 'counts': right_pol[:,1]})
                }
        return result


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


    def window_obs(data: dict, f_min: float, f_max: float) -> None:
        """Window observation data around 21cm line based on frequency range of Bonn data"""
        for info in data.values():
            for pole, df in info.items():
                if 'POL' in pole:
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
                if 'POL' in pole:
                    longitude[pole]['counts'] = df['counts'] / df['counts'].sum() 
                    longitude[pole]['counts'] -= df['counts'].mean()


    def normalise_floor_ref(data:dict) -> None:
        """Normalise and floor reference data counts around 21cm line"""
        for pole, df in data.items():
            if 'POL' in pole:
                data[pole]['counts'] = df['counts'] / df['counts'].sum() 
                data[pole]['counts'] -= df['counts'].mean()


    def get_bt_scale_factor(ref_pol:dict, HI_profile:pd.DataFrame) -> float:
        """Scale the correct polimetry data from the reference spectrum to the HI profile peak 
        to set Brightness Temperature scale"""
        f = HI_profile['BT'].max() / ref_pol['counts'].max()
        return f


    def bt_scale_obs(data:dict, HI_profiles:dict) -> None:
        """Apply brightness temp scale factor to observation data based on HI Profile"""
        for longitude in data.values():
            peak_BT = HI_profiles[longitude['GAL_LONG']]['BT'].max()
            for pole in longitude.keys():
                if 'POL' in pole:
                    factor = peak_BT / longitude[pole]['counts'].max()
                    longitude[pole]['counts'] *= factor
        

    def bt_scale_ref(data:dict, factor:float) -> None:
        """Apply brightness temp scale factor to the single reference dataset"""
        for pole in data.keys():
            if 'POL' in pole:
                data[pole]['counts'] *= factor


    def get_LSR_correction(header) -> float:
        """Wrapper for Wolfgang Herrmann, Astropeiler Stockert code which
        gets observer's velocity relative to Local Standard of Rest"""
        # print(header['DATE-OBS'])
        # print(Time(header['DATE-OBS'], scale='utc'))
        # Extracting the observation times from the FITS header
        date_obs = Time(header['DATE-OBS'], scale='utc')
        date_end = Time(header['DATE-END'], scale='utc')
        avg_time = date_obs + 0.5 * (date_end - date_obs)
        _, vlsr = vlsr_calc(obs_lon=header['SITELONG'], 
                        obs_lat=header['SITELAT'], 
                        obs_ht=header['SITEELEV'], 
                        ra=header['RA'],
                        dec=header['DEC'], 
                        time=date_obs
                        )
        return vlsr 


    def correct_obs_vlsr(data:dict, f0:float) -> None:
        """Applies LSR relative velocity frequenc correction to observation data"""
        for longitude in data.values():
            vslr = get_LSR_correction(longitude['HEADER']).si.value
            print(vslr)
            delta_f = (f0 * vslr) / const.c.value
            for pole in longitude.keys():
                if 'POL' in pole:
                    longitude[pole]['frequency'] -= delta_f


    def correct_ref_vlsr(data:dict, vlsr:float, f0:float) -> None:
        """Applies LSR relative velocity frequenc correction to reference data"""
        delta_f = f0 * (vlsr) / const.c.value
        for pole in data.keys():
            if 'POL' in pole:
                data[pole]['frequency'] -= delta_f

    



