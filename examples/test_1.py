import gpuRIR
import numpy as np
import soundfile as sf
# 生成(min, max)之间的一个随机值
class Parameter:
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self.random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else:
            raise Exception(
                'Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    def getvalue(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value
class GpuRirDemo:
    def __init__(self, room_sz, t60, beta, fs, array_pos):
        self.room_sz = room_sz
        self.t60 = t60
        self.beta = beta
        self.fs = fs
        self.array_pos = array_pos
    def simulate(self):
        if self.t60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1, 1, 1]
        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(15, self.t60)
            Tmax = gpuRIR.att2t_SabineEstimator(60, self.t60)
            if self.t60 < 0.15: Tdiff = Tmax
            nb_img = gpuRIR.t2n(Tdiff, self.room_sz)
        # mic position
        mic_pos = np.array(((-0.079, 0.000, 0.000),
                            (-0.079, -0.009, 0.000),
                            (0.079, 0.000, 0.000),
                            (0.079, -0.009, 0.000)))
        # 阵列中心的坐标
        array_pos = self.array_pos * self.room_sz
        mic_pos = mic_pos + array_pos
        # 声源位置，这里给定一个声源
        source_pos = np.random.rand(3) * self.room_sz
        # 生成RIR
        RIR = gpuRIR.simulateRIR(
            room_sz=self.room_sz,
            beta=self.beta,
            nb_img=nb_img,
            fs=self.fs,
            pos_src=np.array([source_pos]),
            pos_rcv=mic_pos,
            Tmax=Tmax,
            Tdiff=Tdiff,
            mic_pattern='omni'
        )
        # 读取语音
        y, sr = sf.read('source_signal_2.wav')
        # 生成多通道语音
        mic_sig = gpuRIR.simulateTrajectory(y, RIR, fs=self.fs)
        sf.write('filter.wav', mic_sig, sr)
test_code = GpuRirDemo(
    room_sz=Parameter([3, 3, 2.5], [4, 5, 3]).getvalue(), # 此时得到随机得到[3,3,2.5]~[4,5,3]之间的一个房间尺寸
    t60=Parameter(0.2, 1.0).getvalue(), # 0.2s~1.0s之间的一个随机值
                       beta=Parameter([0.5]*6, [1.0]*6).getvalue(), # 房间反射系数
                       array_pos=Parameter([0.1, 0.1, 0.1], [0.9, 0.9, 0.5]).getvalue(),# 比例系数，实际的array_pos为 array_pos * room_sz
                       fs=44100)
test_code.simulate()

