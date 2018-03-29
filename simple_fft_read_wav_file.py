
""" ref: https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    http://stackoverflow.com/questions/3637350/how-to-write-stereo-wav-files-in-python
    http://www.sonicspot.com/guide/wavefiles.html
    Now we read in the data, FFT it, find the coefficient with maximum power, and
    find the corresponding fft frequency, and then convert to Hertz:


    Note: using ALSA on a Linux Operating System, use this command to record your own .wav file
    ## record for 3 seconds, format: 16Bit Little Endian, Mono, 44100 sampling rate, file_name.wav
        $ arecord -d 3 -f S16_LE -c1 -r44100 test.wav
"""

import math
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt

fname = "test.wav"
wav_file = wave.open(fname, 'r')
fs = wav_file.getframerate()            # CD Quality Audio, sampling frequency / sampling rate (avg # of samples in one second)  (44100.0 ?)
print("fs: ", fs)
nchannels = wav_file.getnchannels()     # returns number of audio channels (1 for mono, 2 for stereo)
print("nchannels: ", nchannels)
sampwidth = wav_file.getsampwidth()     # returns sample width in bytes (2)
print("sampwidth: ", sampwidth)
nframes = wav_file.getnframes()         # returns number of audio frames
print("nframes: ", nframes)
comptype = wav_file.getcomptype()       # returns compression type ('NONE' for linear samples)
print("comptype: ", comptype)

N = nframes                             # Number of data sampling points
data = wav_file.readframes(N)
wav_file.close()
data = struct.unpack('{n}h'.format(n=N), data)       # What does this line do?
data = np.array(data)

# Compute the one-dimensional discrete Fourier Transform.
Xf_mag = np.abs(np.fft.fft(data))

# Each index of the Xf_mag array will then contain the amplitude of a frequency
# bin whose frequency is given by index * fs/len(Xf_mag).
freqs = np.fft.fftfreq(len(Xf_mag), d=1.0/fs)
# freqs = np.fft.fft(Xf_mag)     # fail

# Find the peak in the coefficients
idx = np.argmax(np.abs(Xf_mag))
# Reminder: frequency (Hz) = abs(fft_freq * frame_rate).
peak_freq_in_hertz = abs(freqs[idx])
print("index: ",idx,"    Peak Frequency (Hz) :", peak_freq_in_hertz)

half_Xf_mag=Xf_mag[:int(N/2)]
print("Size of positive FFT results array: ", half_Xf_mag.size)
bins = np.zeros(16)
bin_size = int(fs/16) # 2756 ??
print("bin size: ", bin_size)
for i in range(0, 16):
    result =  np.average(half_Xf_mag[1+i*bin_size:(i+1)*bin_size])
    bins[i] = result
print (bins)

plt.close('all')
plt.figure(num='FFT Analysis Project')
ax1 = subplot(3,2,1)
plt.plot(data[:200])
# plt.plot(data)
plt.title('Raw Data (200 points)')

ax2 = subplot(3,2,2)
plt.plot(Xf_mag)
plt.title('1D Discrete Fourier Transform (DFT)')

ax3 = subplot(3,2,3)
plt.plot(freqs)
plt.title('DFT Frequencies')

ax4 = subplot(3,2,4)
# plt.semilogy(freqs, Xf_mag)  # total fail
plt.plot(freqs[:int(N/2)], Xf_mag[:int(N/2)])
plt.title('FFT Analysis')

# ax5 = subplot(3,2,5)
# plt.semilogy(bins,'bo-')             # Only required when data is wacky...
# plt.title("FFT Bins Log Y Scale")

ax6 = subplot(3,2,6)
plt.plot(bins, 'go-')
plt.title("FFT Bins Uniform Scale")

plt.tight_layout()
plt.show()
