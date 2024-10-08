#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a moving source with a spherical microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal_spherical_moving.wav' with the multi-channel recording simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
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
