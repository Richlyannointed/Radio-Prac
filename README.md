# Galactic Rotation Velocity Practical

## Project Overview

This project involves analyzing HI (neutral hydrogen) spectra obtained from the Tony Fairall Teaching Observatory’s sister radio telescope located in Tallarook, Australia. The primary goal is to observe and analyze the galactic plane at various longitudes, calibrate the collected spectra, and determine rotational velocities.

## Key Objectives
1. **Data Collection**: Observing the galactic plane at specific longitudes, recording spectra of hydrogen emissions around the 21 cm line.
2. **Data Stacking**: Stacking multiple observations of the same galactic longitude for a clearer signal.
3. **Calibration**: Using reference spectra to apply a brightness temperature calibration to the observed data.
4. **LSR Correction**: Correcting observed spectra for motion relative to the Local Standard of Rest (LSR).
5. **Radial Velocity Calculation**: Converting frequency data to radial velocity values.
6. **Rotation Curve Analysis**: Using the calibrated spectra to determine rotational velocities at various points along the galactic plane.

## Requirements

Install the necessary libraries using the following command:
```bash
pip install -r requirements.txt
```

## File Structure
helpers.py: This file contains a set of utility functions that perform operations such as reading FITS tables, applying calibration, normalizing data, and correcting for LSR motion.
main.py: The main script where all functions are called. This script runs the complete analysis pipeline from reading observation files to producing the final stacked and calibrated spectra.
