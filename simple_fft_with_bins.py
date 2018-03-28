
"""
    My goal is really to utilize FFT on IoT devices, real time, with minimal latency.  I realized
    that python makes a pretty good scientific prototyping language to get my head in the game,
    and get a basic understanding of what is going on here...  Ultimately I will probably
    move to C/C++ implementations, but for now this python stuff works.  Until this project I
    had just a cursory experience with Python. Spent some time studying up on numpy and
    MatPlotLib for this exercise.

    Reference:
    https://stackoverflow.com/questions/49268647/creating-an-amplitude-vs-frequency-spectrogram-of-an-audio-file-in-python
    (I really like the test signal setup, with visible peak, modulation and white noise)
    see also:
    https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
    (I like the wav file stuff on this posting)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly modulated
    around 3kHz, corrupted by white noise of exponentially decreasing magnitude sampled at 10 kHz.
    Note to self... obviously I added a couple more signals...

    ref on sampling rate: https://en.wikipedia.org/wiki/Sampling_(signal_processing)

    Open Issues: This code is rather fragile for the calculations of fs, N and the number of bins.  All of those data elements
    are entered manually, but probably should be utilized as pure variables.  Desired Nyquist frequency is 22,050 hz.  The numbers
    chosen for fs and N work well.  16 bins for quick FFT output makes sense for iot devices.  Less is more...
    Obviously, the validation for this setup is that the FFT/bin plot strongly resembles the FFT/hz plot.
"""

import numpy as np
import matplotlib.pyplot as plt

fs = 44100      # CD Quality Audio, sampling frequency / sampling rate (avg # of samples in one second)
N = 88200       # Number of data sampling points  originally 1e5
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
xx = carrier + carrier2 + carrier3 + noise

# Compute the one-dimensional discrete Fourier Transform.
Xf_mag = np.abs(np.fft.fft(xx))

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
plt.plot(xx[:200])
# plt.plot(xx)
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
