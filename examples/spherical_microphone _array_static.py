#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a static source with a spherical microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal_spherical.wav' with the multi-channel recording simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import gpuRIR

# Activate Mixed Precision to ensure higher precision
gpuRIR.activateMixedPrecision(False)

# Read the audio file
fs, source_signal = wavfile.read('source_signal_1.wav')

# If the audio has multiple channels, convert it to mono
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Define room dimensions and source position
room_sz = [5, 6, 3]  # Room dimensions [m]
source_pos = np.array([2.5, 3.0, 1.5])  # Static source position (near the center of the room)

# Generate the spherical microphone array positions (assuming 32 microphones)
def spherical_mic_array(radius, num_mics, center_pos):
    """ Generate 3D coordinates for microphones on a spherical surface and place them at the specified center position """
    indices = np.arange(0, num_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_mics)  # Latitude angle
    theta = np.pi * (1 + 5**0.5) * indices      # Longitude angle

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Shift microphone positions to the array center
    mic_positions = np.vstack([x, y, z]).T + center_pos
    return mic_positions

# Define microphone array radius and number of microphones
mic_radius = 0.1  # Radius of the spherical microphone array [m]
num_mics = 32  # Number of microphones

# Ensure the microphone array center is near the center of the room, avoiding array exceeding room bounds
array_center_pos = np.array([2.5, 3.0, 1.5])  # Microphone array center (placed at the room's center)

# Generate spherical microphone array coordinates
mic_positions = spherical_mic_array(mic_radius, num_mics, array_center_pos)

# Define room reverberation time and attenuation parameters
T60 = 0.6  # Reverberation time [seconds]
att_diff = 15.0  # Attenuation [dB] when the diffuse reverberation model starts
att_max = 60.0  # Maximum attenuation [dB]

# Calculate reflection coefficients and simulation time
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Estimate reflection coefficients
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # Time when the diffuse reverberation model starts
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # Maximum simulation time
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # Calculate the number of image sources

# Generate Room Impulse Response (RIR)
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, np.array([source_pos]), mic_positions, nb_img, Tmax, fs,
    Tdiff=Tdiff, mic_pattern="omni"  # Using omnidirectional microphones
)

# Convolve the source signal with the RIR to generate multi-channel microphone signals
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Save the simulation result as a WAV file
wavfile.write('filtered_signal_spherical_static.wav', fs, filtered_signal)

# Plot the waveform from the first microphone in the array
plt.plot(filtered_signal[:, 0])  # Plot the signal from the first microphone
plt.title("Spherical Microphone Array - First Mic Signal (Static Source)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
