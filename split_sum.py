import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def fft_split(data_x, data_y):
    freq_spectre = fft(data_y)

    fs = int(1 / np.mean(np.diff(data_x)))
    
    frequencies = fftfreq(len(data_x), 1 / fs)

    freq_spectre = np.abs(freq_spectre[:len(frequencies) // 2])
    
    frequencies = frequencies[:len(frequencies) // 2]

    peaks, peaks_dict = find_peaks(freq_spectre, height=10)

    peaks_data = peaks_dict['peak_heights']

    max1, max2 = 0, 0
    max1i, max2i = -1, -1
    for i in range(len(peaks_data)):
        if peaks_data[i] > max1:
            max1 = peaks_data[i]
            max1i = peaks[i]
        elif peaks_data[i] > max2:
            max2 = peaks_data[i]
            max2i = peaks[i]

    if max1 - max2 > max2:
        max2i = max1i
    elif max2 - max1 > max1:
        max1i = max2i

    return frequencies[max1i], frequencies[max2i]

    

location = "./data/sum"
filename = "DS0007.csv"
path = location + '/' + filename

pd.options.display.max_rows = 10000

csv_data = pd.read_csv(path, skiprows=25, names=["Time", "Signal"], usecols=[0, 1])
signal_x = [float(elem[0]) for elem in csv_data.values]
signal_y = [float(elem[1]) for elem in csv_data.values]

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(signal_x, signal_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, В")
plt.title("Оригинальный сигнал 1 + 2")
plt.legend()
plt.grid()

freq_1, freq_2 = fft_split(signal_x, signal_y)

signal_1_y = [np.sin(2 * np.pi * freq_1 * i) for i in signal_x]

plt.subplot(3, 1, 2)
plt.plot(signal_x, signal_1_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, В")
plt.title("Сигнал 1")
plt.legend()
plt.grid()

signal_2_y = [np.sin(2 * np.pi * freq_2 * i) for i in signal_x]

plt.subplot(3, 1, 3)
plt.plot(signal_x, signal_2_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, В")
plt.title("Сигнал 2")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
