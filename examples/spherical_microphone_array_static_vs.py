#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating and visualizing a static source with a spherical microphone array in 3D.
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

# Step 2: 定义房间尺寸和声源位置
room_sz = [5, 6, 3]  # 房间的尺寸 [米]
source_pos = np.array([1.5, 2.0, 0.5])  # 声源的静止位置（房间中央附近）

# Step 3: 生成球形麦克风阵列的位置（假设有 32 个麦克风）
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

# 麦克风阵列半径和数量
mic_radius = 0.1  # 球形麦克风阵列的半径 [米]
num_mics = 32  # 麦克风的数量

# Step 4: 确保麦克风阵列中心在房间的中央附近，避免阵列超出房间
array_center_pos = np.array([2.5, 3.0, 1.5])  # 麦克风阵列的中心（放置在房间中央）

# 生成球形麦克风阵列的坐标
mic_positions = spherical_mic_array(mic_radius, num_mics, array_center_pos)

# Step 5: 定义房间的混响时间和衰减参数
T60 = 0.6  # 混响时间 [秒]
att_diff = 15.0  # 开始使用扩散混响模型时的衰减 [dB]
att_max = 60.0  # 最大衰减 [dB]

# Step 6: 计算反射系数和模拟时间
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # 估算反射系数
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # 扩散混响模型的开始时间
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # 最大模拟时间
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # 计算图像源法中的反射次数

# Step 7: 生成房间脉冲响应 (RIR)
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, np.array([source_pos]), mic_positions, nb_img, Tmax, fs,
    Tdiff=Tdiff, mic_pattern="omni"  # 使用全向性麦克风
)

# Step 8: 将声源信号与 RIR 卷积，生成多通道麦克风接收信号
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Step 9: 保存模拟结果为 WAV 文件
wavfile.write('filtered_signal_spherical_static.wav', fs, filtered_signal)

# Step 10: 绘制麦克风阵列中一个麦克风接收到的信号波形
plt.plot(filtered_signal[:, 0])  # 绘制第一个麦克风的信号
plt.title("Spherical Microphone Array - First Mic Signal (Static Source)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Step 11: 在3D图中展示房间、声源和球形麦克风阵列

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

def plot_source(ax, source_pos):
    """ 绘制静止声源 """
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='o', label="Source", s=100)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制房间
plot_room(ax, room_sz)

# 绘制球形麦克风阵列
plot_mic_array(ax, mic_positions)

# 绘制静止声源
plot_source(ax, source_pos)

# 调整视角，确保声源和麦克风阵列的相对位置正确
ax.view_init(elev=30, azim=45)  # 调整仰角和方位角

# 保持 X, Y, Z 轴的比例一致
ax.set_box_aspect([1, 1, 1])

# 设置 X, Y, Z 轴的标签
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
ax.set_zlabel('Z axis [m]')

# 设置 X, Y, Z 轴的范围
ax.set_xlim([0, room_sz[0]])
ax.set_ylim([0, room_sz[1]])
ax.set_zlim([0, room_sz[2]])
ax.legend()

# 显示图形
plt.title("Room with Spherical Microphone Array and Static Source")
plt.show()
