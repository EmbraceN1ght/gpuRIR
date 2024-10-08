#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating and visualizing a static source with a spherical microphone array in 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gpuRIR

# Activate Mixed Precision for higher precision
gpuRIR.activateMixedPrecision(False)

# Read the audio file
fs, source_signal = wavfile.read('source_signal_1.wav')

# If the audio has multiple channels, convert it to mono
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Define the room size and source position
room_sz = [5, 6, 3]  # Room dimensions [m]
source_pos = np.array([1.5, 2.0, 0.5])  # Position of the static source (near the center of the room)

# Generate the positions of the spherical microphone array (assuming 32 microphones)
def spherical_mic_array(radius, num_mics, center_pos):
    """ Generate 3D coordinates for microphones on a spherical surface and place them at the specified center position """
    indices = np.arange(0, num_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_mics)  # Latitude angle
    theta = np.pi * (1 + 5**0.5) * indices      # Longitude angle

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Shift the microphone positions to the center of the array
    mic_positions = np.vstack([x, y, z]).T + center_pos
    return mic_positions

# Radius and number of microphones in the array
mic_radius = 0.1  # Radius of the spherical microphone array [m]
num_mics = 32  # Number of microphones

# Ensure the microphone array is centered in the room, avoiding exceeding room bounds
array_center_pos = np.array([2.5, 3.0, 1.5])  # Center position of the microphone array (placed near the room's center)

# Generate the coordinates of the spherical microphone array
mic_positions = spherical_mic_array(mic_radius, num_mics, array_center_pos)

# Define the room's reverberation time and attenuation parameters
T60 = 0.6  # Reverberation time [seconds]
att_diff = 15.0  # Attenuation [dB] when the diffuse reverberation model starts
att_max = 60.0  # Maximum attenuation [dB]

# Calculate reflection coefficients and simulation time
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Estimate reflection coefficients
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # Time when the diffuse reverberation model starts
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # Maximum simulation time
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # Calculate the number of image sources

# Generate the Room Impulse Response (RIR)
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

# Plot the room, source, and spherical microphone array in 3D

def plot_room(ax, room_sz):
    """ Draw the boundaries of the room """
    x, y, z = room_sz
    vertices = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                         [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [[vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [3, 0, 4, 7]]]
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.1))

def plot_mic_array(ax, mic_positions):
    """ Plot the spherical microphone array """
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', marker='o', label="Microphones", s=50)

def plot_source(ax, source_pos):
    """ Plot the static source """
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='o', label="Source", s=100)

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the room
plot_room(ax, room_sz)

# Plot the spherical microphone array
plot_mic_array(ax, mic_positions)

# Plot the static source
plot_source(ax, source_pos)

# Adjust the view to ensure the relative positions of the source and microphone array are correct
ax.view_init(elev=30, azim=45)  # Adjust elevation and azimuth

# Keep the aspect ratio of the X, Y, Z axes consistent
ax.set_box_aspect([1, 1, 1])

# Set the labels for the X, Y, Z axes
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
ax.set_zlabel('Z axis [m]')

# Set the range for the X, Y, Z axes
ax.set_xlim([0, room_sz[0]])
ax.set_ylim([0, room_sz[1]])
ax.set_zlim([0, room_sz[2]])
ax.legend()

# Display the figure
plt.title("Room with Spherical Microphone Array and Static Source")
plt.show()
