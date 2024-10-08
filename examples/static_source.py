#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a **static source** with a microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal.wav' with the stereo recording simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
