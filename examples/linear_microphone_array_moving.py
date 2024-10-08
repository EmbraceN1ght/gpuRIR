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

# Step 1: 读取音频文件
fs, source_signal = wavfile.read('source_signal_1.wav')

# 如果是多通道音频，将其转换为单通道
if len(source_signal.shape) > 1:
    source_signal = np.mean(source_signal, axis=1)

# Step 2: 定义房间尺寸、移动声源的轨迹和麦克风阵列
room_sz = [3, 4, 2.5]  # 房间的尺寸 [米]
traj_pts = 64  # 声源移动的轨迹点数
pos_traj = np.tile(np.array([0.0, 3.0, 1.0]), (traj_pts, 1))  # 初始化轨迹点
pos_traj[:, 0] = np.linspace(0.1, 2.9, traj_pts)  # 沿 x 轴移动声源

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
    room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs,
    Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
)

# Step 6: 将声源信号与RIR卷积，生成多通道麦克风接收信号
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Step 7: 保存模拟结果
wavfile.write('filtered_signal_moving.wav', fs, filtered_signal)

# Step 8: 绘制模拟的麦克风接收信号
plt.plot(filtered_signal)
plt.title('Simulated Microphone Signals for Moving Source')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# Step 9: 在3D中绘制房间、麦克风阵列和声源轨迹
def plot_room(ax, room_sz):
    """ 绘制房间的边界 """
    x, y, z = room_sz
    vertices = np.array([[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
                         [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]])
    edges = [[vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [3, 0, 4, 7]]]
    ax.add_collection3d(Poly3DCollection(edges, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.1))

def plot_mic_array(ax, pos_rcv):
    """ 绘制麦克风阵列 """
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], c='b', marker='^', label="Microphones", s=100)
    ax.plot(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2], color='blue', linewidth=2)

def plot_source_trajectory(ax, pos_traj):
    """ 绘制移动声源的轨迹 """
    ax.scatter(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], c='r', marker='o', label="Source", s=50)
    ax.plot(pos_traj[:, 0], pos_traj[:, 1], pos_traj[:, 2], color='red', linestyle='dashed')

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制房间
plot_room(ax, room_sz)

# 绘制麦克风阵列
plot_mic_array(ax, pos_rcv)

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
plt.title("Room with Linear Microphone Array and Moving Source")
plt.show()
