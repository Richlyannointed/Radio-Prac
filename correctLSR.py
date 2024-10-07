#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlsr_calc is a routine to do LSR corrections for radio astronomy using Astropy
In addition to the routine itself some examples of its usage are given
Author: Wolfgang Herrmann, Astropeiler Stockert
Version 1.1, August 7th, 2021
"""

#Required imports
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.coordinates import ICRS, LSR

"""
Here comes the routine to do the calculation
Required inputs:
obs_lon: Geographic longitude of the observatory in decimal degrees
obs_lat: Geographic latitude of the observatory in decimal degrees
obs_ht: Heigt of the observatory in meters
ra: Right ascension of the observed sky location in decimal hours
dec: Declination of the observed sky location in decimal degrees
time: Time of observation in UTC (Astropy format)
Note that there is a specific definiton of the peculiar motion of the sun
overriding the default.
This routine returns two values: The velocity of the observer with respect to
the solar system barycenter and the velocity of the observer with respect to
the local standard of rest.
"""
def vlsr_calc(obs_lon, obs_lat, obs_ht, ra, dec, time):
    obsloc = EarthLocation(lon=obs_lon*u.deg, lat=obs_lat*u.deg, height=obs_ht*u.m)
    skycoord = SkyCoord((ra*15.)*u.deg, dec*u.deg)
    vbary = skycoord.radial_velocity_correction(kind='barycentric', obstime=time, location=obsloc)  
    my_observation = ICRS((ra*15.)*u.deg, dec*u.deg, pm_ra_cosdec=0*u.mas/u.yr, pm_dec=0*u.mas/u.yr, \
    radial_velocity=vbary, distance = 1*u.pc)
    vlsr = my_observation.transform_to(LSR(v_bary=(10.27,15.32,7.74)*u.km/u.s)).radial_velocity
    return vbary,vlsr

def main():

    #Now let's make an example how to use it
        
    #Define the observatory location
    #This example uses the coordinates of the Stockert observatory
    obs_lon = 6.72194444
    obs_lat = 50.569444
    obs_ht  = 434.0

    #Define the sky location observed
    ra = 0.43
    dec = 62.73

    #Define the time of observation
    #If you want to take the current time:
    now = datetime.utcnow()
    time=Time(now, scale='utc')

    #Here is an example on how to give an explicit time
    #Uncomment of you want to use this example
    #mytime = '2021-08-07T00:00:00'
    #time=Time(mytime, scale='utc')


    #Throw this into the calculation and print
    velocities = vlsr_calc(obs_lon, obs_lat, obs_ht, ra, dec, time)
    print ("Example 1")
    print ("VLSR: ",velocities[1])
    print ("Barycentric velocity:", velocities[0])

    #You will notice that it is printed with units
    #This is because the velocities are returned as Astropy entities.
    #You may want to have this as simple floating point numbers:
    vlsr = velocities[1].value
    vbary = velocities[0].value/1000.
    print ("Same as pure numbers in km/s: ",vlsr,vbary)
    print("***")

    #Your may want to use this to convert frequencies. For this we need the
    #speed of light and the rest frequency.
    c=299792.458    #speed of light in km/s
    rest_frequency = 1420.405 #Hydrogen line in MHz

    #This example calculates the sky frequency of the hydrogen rest frequency.
    #You could use this to adjust your receiver frequency to record spectra in the LSR frame
    sky_frequency = rest_frequency *(1-vlsr/c)
    print ("Example 3")
    print ("Sky frequency of a hydrogen line at rest:", sky_frequency)
    print("***")

    #Another example is to determine the LSR velocity of a line observed without correction:
    #So let's assume you have some hydrogen line at 1420.412 MHz
    #observed_frequency=sky_frequency
    observed_frequency = 1420.412
    LSR_velocity=c*(1-(observed_frequency/rest_frequency))-vlsr
    print ("Example 4")
    print("LSR velocity:", LSR_velocity)

if __name__ =='__main__':
    main()