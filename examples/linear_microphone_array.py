#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a **static source** with a microphone array,
and plotting the room, microphones, and source in 3D using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gpuRIR
gpuRIR.activateMixedPrecision(False)

# Step 1: 读取音频文件
fs, source_signal = wavfile.read('source_signal_1.wav')

# 如果是多通道音频，将其转换为单通道
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Step 2: 定义房间尺寸、声源位置和麦克风阵列
room_sz = [3, 4, 2.5]  # 房间的尺寸 [米]
source_pos = np.array([1.5, 3.0, 1.0])  # 静止声源的位置 (x, y, z)

# 定义麦克风阵列的位置
nb_rcv = 2  # 麦克风的数量
pos_rcv = np.array([[1.4, 1, 1.5], [1.6, 1, 1.5]])  # 两个麦克风的位置
orV_rcv = np.array([[-1, 0, 0], [1, 0, 0]])  # 麦克风的方向向量
mic_pattern = "card"  # 心形指向性麦克风

# Step 3: 定义混响时间和衰减参数
T60 = 0.6  # 混响时间（0.6 秒）
att_diff = 15.0  # 15 dB 时开始使用扩散模型
att_max = 60.0  # 60 dB 时停止模拟

# Step 4: 计算反射系数和模拟时间
beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # 估算反射系数
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)  # 扩散模型生效时间
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)  # 最大模拟时间
nb_img = gpuRIR.t2n(Tdiff, room_sz)  # 计算图像源数量

# Step 5: 生成房间脉冲响应 (RIR)
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, np.array([source_pos]), pos_rcv, nb_img, Tmax, fs,
    Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
)

# Step 6: 将声源信号与RIR卷积，生成多通道麦克风接收信号
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Step 7: 保存模拟结果
wavfile.write('filtered_signal_static.wav', fs, filtered_signal)

# Step 8: 绘制结果
plt.plot(filtered_signal)
plt.title("Simulated Microphone Signals for Static Source")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Step 9: 在3D中绘制房间、麦克风阵列和声源

def plot_room(ax, room_sz):
    """ 绘制房间的边界 """
    x, y, z = room_sz
    r = [0, x], [0, y], [0, z]
    vertices = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0], [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [[vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [3, 0, 4, 7]]]
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.1))

def plot_mic_array(ax, pos_rcv):
    """ 绘制麦克风阵列 """
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], c='b', label="Microphones", s=100)
    ax.plot(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], color='blue', linewidth=2)

def plot_source(ax, source_pos):
    """ 绘制声源位置 """
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], c='r', marker='o', label="Source", s=100)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制房间
plot_room(ax, room_sz)

# 绘制麦克风阵列，使用三角形标记麦克风
plot_mic_array(ax, pos_rcv)

# 绘制声源，使用圆形标记声源
plot_source(ax, source_pos)

# 调整视角，elev 是仰角，azim 是方位角
ax.view_init(elev=30, azim=45   )

# 设置 X, Y, Z 轴的比例，使其一致
ax.set_box_aspect([1, 1, 1])  # 保持 X, Y, Z 轴的比例为 1:1:1

# 设置图形参数
ax.set_xlabel('X axis [m]')
ax.set_ylabel('Y axis [m]')
ax.set_zlabel('Z axis [m]')
ax.set_xlim([0, room_sz[0]])
ax.set_ylim([0, room_sz[1]])
ax.set_zlim([0, room_sz[2]])
ax.legend()

# 显示图形
plt.title("Room with Linear Microphone Array and Static Source")
plt.show()

