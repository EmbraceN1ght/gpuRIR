#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating and visualizing a moving source with a spherical microphone array in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gpuRIR

# 激活 Mixed Precision，确保精度较高
gpuRIR.activateMixedPrecision(False)

# Step 1: 读取音频文件
fs, source_signal = wavfile.read('source_signal_1.wav')

# 如果是多通道音频，将其转换为单通道
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Step 2: 定义房间尺寸和麦克风阵列的位置
room_sz = [5, 6, 3]  # 房间的尺寸 [米]

# 球形麦克风阵列的生成函数
def spherical_mic_array(radius, num_mics, center_pos):
    """ 在球面上生成麦克风的三维坐标并放置在指定中心位置 """
    indices = np.arange(0, num_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_mics)  # 纬度角
    theta = np.pi * (1 + 5**0.5) * indices      # 经度角

    # 将球面坐标转换为笛卡尔坐标
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # 将麦克风位置平移到阵列中心点
    mic_positions = np.vstack([x, y, z]).T + center_pos
    return mic_positions

# Step 3: 确保麦克风阵列中心在房间的中央，避免阵列超出房间
mic_radius = 0.1  # 球形麦克风阵列的半径 [米]
num_mics = 32  # 麦克风的数量
array_center_pos = np.array([2.5, 3.0, 1.5])  # 麦克风阵列的中心

# 生成球形麦克风阵列的坐标
mic_positions = spherical_mic_array(mic_radius, num_mics, array_center_pos)

# Step 4: 定义声源的移动轨迹
traj_pts = 64  # 轨迹点数
pos_traj = np.tile(np.array([0.0, 3.0, 1.0]), (traj_pts, 1))
pos_traj[:, 0] = np.linspace(0.5, 4.5, traj_pts)  # 沿 x 轴移动，确保不出房间

# Step 5: 定义房间的混响时间和衰减参数
T60 = 0.6  # 混响时间 [秒]
att_diff = 15.0  # 开始使用扩散混响模型时的衰减 [dB]
att_max = 60.0  # 最大衰减 [dB]

# Step 6: 计算反射系数和模拟时间
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # 估算反射系数
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # 扩散混响模型的开始时间
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # 最大模拟时间
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # 计算图像源法中的反射次数

# Step 7: 生成每个轨迹点上的房间脉冲响应 (RIR)
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, pos_traj, mic_positions, nb_img, Tmax, fs,
    Tdiff=Tdiff, mic_pattern="omni"  # 使用全向性麦克风
)

# Step 8: 将声源信号与移动轨迹上的 RIR 卷积，生成多通道麦克风接收信号
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Step 9: 保存模拟结果为 WAV 文件
wavfile.write('filtered_signal_spherical_moving.wav', fs, filtered_signal)

# Step 10: 绘制麦克风阵列中一个麦克风接收到的信号波形
plt.plot(filtered_signal[:, 0])  # 绘制第一个麦克风的信号
plt.title("Spherical Microphone Array - First Mic Signal (Moving Source)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Step 11: 在3D图中展示房间、移动声源和球形麦克风阵列

def plot_room(ax, room_sz):
    """ 绘制房间的边界 """
    x, y, z = room_sz
    vertices = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                         [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [[vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [3, 0, 4, 7]]]
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.1))

def plot_mic_array(ax, mic_positions):
    """ 绘制球形麦克风阵列 """
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='b', marker='o', label="Microphones", s=50)

def plot_source_trajectory(ax, pos_traj):
    """ 绘制移动声源的轨迹 """
    ax.scatter(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], c='r', marker='o', label="Source", s=50)
    ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], color='red', linestyle='dashed')

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制房间
plot_room(ax, room_sz)

# 绘制球形麦克风阵列
plot_mic_array(ax, mic_positions)

# 绘制声源移动轨迹
plot_source_trajectory(ax, pos_traj)

# 调整视角，确保声源和麦克风阵列的相对位置正确
ax.view_init(elev=30, azim=45)  # 调整仰角和方位角

# 保持 X, Y, Z 轴的比例一致
ax.set_box_aspect([1, 1, 1])

# 设置 X, Y, Z 轴的标签，确保理解方向正确
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
ax.set_zlabel('Z axis [m]')

# 设置 X, Y, Z 轴的范围
ax.set_xlim([0, room_sz[0]])
ax.set_ylim([0, room_sz[1]])
ax.set_zlim([0, room_sz[2]])
ax.legend()

# 显示图形
plt.title("Room with Spherical Microphone Array and Moving Source")
plt.show()

# Step 1: 选择一个声源和接收器
rir = RIRs[0, 0, :]  # 选择第1个声源和第1个麦克风的RIR

# Step 2: 获取时间轴
fs = 16000  # 假设采样频率为 16 kHz
time_axis = np.arange(len(rir)) / fs  # 生成时间轴（以秒为单位）

# Step 3: 绘制 RIR 波形
plt.figure(figsize=(10, 4))
plt.plot(time_axis, rir)
plt.title('Room Impulse Response (RIR)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()