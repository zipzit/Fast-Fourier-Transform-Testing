""" ref: https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    http://stackoverflow.com/questions/3637350/how-to-write-stereo-wav-files-in-python
    http://www.sonicspot.com/guide/wavefiles.html
"""
import math
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt


freq1 = 3000.0
freq2 = 15000.0
freq3 = 20000.0
N = 88200              # Number of data sampling points
fname = "test.wav"
fs = 44100.0           # CD Quality Audio, sampling frequency / sampling rate (avg # of samples in one second)
amp = 64000.0
nchannels = 1          # number of audio channels (1 for mono, 2 for stereo)
sampwidth = 2          # sample width in bytes
framerate = int(fs)
nframes = N
comptype = "NONE"      # compression type ('NONE' for linear samples)
compname = "not compressed"    # human-readable version of compression type ('not compressed' for linear samples)
# ---Simple Data Generator-------------------------------------------------------
# data = [math.sin(2 * math.pi * freq1 * (x / fs))
        # for x in range(N)]

# ---Complex Data Generator-----------------------------------------------------
amp = 2 * np.sqrt(2)        # Originally set to 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
# time = np.linspace(0.0, 1.0, 44100)
mod = 400*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)       # 3000 hz
carrier2 = amp * np.sin(2*np.pi*15e3*time + mod)     # 15000 hz
carrier3 = amp * np.sin(2*np.pi*20e3*time + mod)     # 20000 hz
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
data = carrier + carrier2 + carrier3 + noise

wav_file = wave.open(fname, 'w')
wav_file.setparams(
    (nchannels, sampwidth, framerate, nframes, comptype, compname))
for v in data:
    wav_file.writeframes(struct.pack('h', int(v * amp / 2)))
wav_file.close()
