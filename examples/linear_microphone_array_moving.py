#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating and visualizing a moving source with a microphone array in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gpuRIR
gpuRIR.activateMixedPrecision(False)

# Read the audio file
fs, source_signal = wavfile.read('source_signal_1.wav')

# If the audio has multiple channels, convert it to mono
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Define the room dimensions, source trajectory, and microphone array
room_sz = [3, 4, 2.5]  # Room dimensions [m]
traj_pts = 64  # Number of trajectory points
pos_traj = np.tile(np.array([0.0, 3.0, 1.0]), (traj_pts, 1))  # Initialize trajectory points
pos_traj[:, 0] = np.linspace(0.1, 2.9, traj_pts)  # Source moves along the x-axis

# Define microphone array positions
nb_rcv = 2  # Number of microphones
pos_rcv = np.array([[1.4, 1, 1.5], [1.6, 1, 1.5]])  # Positions of two microphones
orV_rcv = np.array([[-1, 0, 0], [1, 0, 0]])  # Orientation vectors of microphones
mic_pattern = "card"  # Cardioid microphone pattern

# Define reverberation time and attenuation parameters
T60 = 0.6  # Reverberation time (0.6 seconds)
att_diff = 15.0  # Diffuse model starts at 15 dB
att_max = 60.0  # Maximum attenuation (60 dB)

# Compute reflection coefficients and simulation time
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Estimate reflection coefficients
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # Diffuse model activation time
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # Maximum simulation time
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # Calculate number of image sources

# Generate Room Impulse Response (RIR)
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs,
    Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
)

# Convolve the source signal with RIR to generate multi-channel microphone signals
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Save the simulation results
wavfile.write('filtered_signal_moving.wav', fs, filtered_signal)

# Plot the simulated microphone signals
plt.plot(filtered_signal)
plt.title('Simulated Microphone Signals for Moving Source')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# Plot the room, microphone array, and source trajectory in 3D
def plot_room(ax, room_sz):
    """ Plot the room boundaries """
    x, y, z = room_sz
    vertices = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                         [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [[vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [3, 0, 4, 7]]]
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.1))

def plot_mic_array(ax, pos_rcv):
    """ Plot the microphone array """
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], c='b', marker='^', label="Microphones", s=100)
    ax.plot(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], color='blue', linewidth=2)

def plot_source_trajectory(ax, pos_traj):
    """ Plot the source trajectory """
    ax.scatter(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], c='r', marker='o', label="Source", s=50)
    ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], color='red', linestyle='dashed')

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the room
plot_room(ax, room_sz)

# Plot the microphone array
plot_mic_array(ax, pos_rcv)

# Plot the source trajectory
plot_source_trajectory(ax, pos_traj)

# Adjust view angle, elev is elevation, azim is azimuth
ax.view_init(elev=30, azim=45)

# Set axis aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])

# Set axis labels and limits
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
ax.set_zlabel('Z axis [m]')
ax.set_xlim([0, room_sz[0]])
ax.set_ylim([0, room_sz[1]])
ax.set_zlim([0, room_sz[2]])
ax.legend()

# Show the plot
plt.title("Room with Linear Microphone Array and Moving Source")
plt.show()
