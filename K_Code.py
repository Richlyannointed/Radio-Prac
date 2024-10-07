import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Function to read data from a text file
def read_text_file(filename):
    # Read the file and skip the first lines that contain metadata
    data = np.loadtxt(filename, comments='%')
    frequency = data[:, 2] * 1e6  # Convert frequency from MHz to Hz
    intensity = data[:, 1]  # Assuming this is T_B [K]
    return frequency, intensity


def process_fits_file(fits_file, obsfreq_correction):
    # Open the FITS file
    tab = fits.open(fits_file)[1]

    # Access data and FITS header
    data = tab.data
    header = tab.header

    # Extract necessary columns
    status_column = data['STATUS']
    left_polarimetry_column = data['LEFT_POL']  # Assuming this column contains left polarimetry data

    # Extract necessary header information for frequency calculation
    base_freq = header['BASEFREQ']    # Baseband frequency in Hz
    bndres = header['BNDRES']         # Spectral resolution (Hz)
    # Apply correction to OBSFREQ

    observed_freq_shift = header['OBSFREQ'] - obsfreq_correction
    base_freq -= observed_freq_shift
    # Generate the frequency scale for the 1024 channels
    num_channels = 1024
    frequency_scale = base_freq + np.arange(num_channels) * bndres

    # Extract data where status is 'on' or 'off'
    filtered_left_polarimetry = left_polarimetry_column[(status_column == 'on') | (status_column == 'off')]
        # Combine the extracted left polarimetry data (using mean)
    combined_polarimetry_data = np.mean(filtered_left_polarimetry, axis=0)
    return frequency_scale, combined_polarimetry_data

# Define the corrected observed frequency (1428.75 MHz)
obsfreq_correction = 1428.75e6  # Corrected observed frequency in Hz

# Process each FITS file
fits_files = ["20240919-101005_SPECTRUM-PROJ01-GAL_01#_01#.fits",
              "20240919-101741_SPECTRUM-PROJ01-GAL_02#_01#.fits",
              "20240919-102500_SPECTRUM-PROJ01-GAL_03#_01#.fits"]  # Replace with actual FITS file names

# Initialize lists to store data
all_frequencies = []
all_polarimetry_data = []

for fits_file in fits_files:
    freq_scale, polarimetry_data = process_fits_file(fits_file, obsfreq_correction)
    all_frequencies.append(freq_scale)
    all_polarimetry_data.append(polarimetry_data)

# Combine data from all FITS files (taking the average)
combined_frequency_scale = all_frequencies[0]  # Assuming frequency scale is consistent
combined_polarimetry_data = np.mean(all_polarimetry_data, axis=0)

# Normalize the intensity
#normalized_polarimetry_data = combined_polarimetry_data / np.max(combined_polarimetry_data)

# Define the range for the x-axis (frequency) between 1419 MHz and 1422 MHz
min_freq = 1418e6  # 1419 MHz in Hz
max_freq = 1423e6  # 1422 MHz in Hz
# Filter the data to plot only within the desired frequency range
valid_indices = (combined_frequency_scale >= min_freq) & (combined_frequency_scale <= max_freq)
frequency_scale_filtered = combined_frequency_scale[valid_indices]
combined_polarimetry_filtered = combined_polarimetry_data[valid_indices]
#corresction for LSR
# Correction for LSR
correction = (1420.4496195374156 - 1420.405)*(10*6)

# Convert to velocity
c = 3e5  # speed of light in km/s
f_rest = 1420.405   # rest frequency in MHz
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Frequency vs Scaled Intensity
ax1.plot((frequency_scale_filtered-correction)/ 1e6+0.2, -324+(6.5e-8)*combined_polarimetry_filtered, color='blue', label='Observational data for l=332')
ax1.set_xlabel('Frequency (MHz)', fontsize=12)
ax1.set_ylabel('Brightness Temperature (K)', fontsize=12)
ax1.set_title('Plot of the observational data at l=332 showing the brightness temperature at the peak frequency/velocity', fontsize=14)
ax1.grid(True)
ax1.invert_xaxis()

# Create a twin axis for velocity
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Ensure the limits are the same

# Define the desired velocity ticks
velocity_ticks = np.arange(-600, 601, 100)

# Calculate corresponding frequencies for these velocities using the correct relationship
frequencies_for_velocities = (((velocity_ticks)/c)*f_rest)+f_rest+0.79

# Set the velocity ticks on the velocity axis
ax2.set_xticks(frequencies_for_velocities)
ax2.set_xticklabels(velocity_ticks)
ax2.set_xlabel('Velocity (km/s)', fontsize=12)
ax2.invert_xaxis()

# Read data from the text file
text_file = 'spectrum.txt'
text_frequency, text_intensity = read_text_file(text_file)
r_min=8.5*np.sin(np.radians(360-332))# Find the index of the peak intensity
peak_index = np.argmax(text_intensity)

# Get the peak intensity and corresponding frequency
peak_intensity = text_intensity[peak_index]
peak_frequency = text_frequency[peak_index]

print(f"Peak Intensity: {peak_intensity} K")
print(f"Peak Frequency: {peak_frequency/1e6} MHz")
print(f"R_min= {r_min:.3f}kpc")
#scaling factor for brightness temp is: (9*10**(-6.41))
# Overlay the text file data
ax1.plot(text_frequency / 1e6, text_intensity, color='red', label='Bonn Lab data', linestyle="--")

ax1.axvline(x=peak_frequency/1e6, color='k', linestyle='--', label=f"Peak at {peak_frequency/1e6:.2f} MHz ({-c*((peak_frequency/1e6)-f_rest)/(f_rest):.2f}km/s)")
ax1.axvline(x=1420.75, color='green', linestyle='--', label=f"Max Peak at {1420.75:.2f} MHz ({-c*((1420.75)-f_rest)/(f_rest):.2f}km/s)")
# Display the legend and plot
ax1.legend()
plt.savefig('l332.png',dpi=300,bbox_inches='tight')
plt.show()
