import numpy as np
from numpy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
import pandas as pd

def moving_average_filter(data_x, data_y, window_size=10):
    return data_x, np.convolve(data_y, np.ones(window_size)/window_size, mode='same')

def fft_filter(data_x, data_y, high=10000):
    freq_spectre = fft(data_y)

    frequencies = fftfreq(len(data_x), np.mean(np.diff(data_x)))

    freq_spectre[np.abs(frequencies) > high] = 0
    
    filtered = ifft(freq_spectre).real
    return data_x, filtered

location = "./data/noise"
filename = "DS0004.csv"
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
plt.title("Оригинальный сигнал")
plt.legend()
plt.grid()

window_size = 1000 # размер окна сглаживания
maf_signal_x, maf_signal_y = moving_average_filter(signal_x, signal_y, window_size)

# строим график
plt.subplot(3, 1, 2)
plt.plot(maf_signal_x, maf_signal_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, В")
plt.title("Сглаживание методом скользящего среднего")
plt.legend()
plt.grid()

discard_above = 2000 # 2кГц
fft_signal_x, fft_signal_y = fft_filter(signal_x, signal_y, discard_above)

# строим график
plt.subplot(3, 1, 3)
plt.plot(fft_signal_x, fft_signal_y, label="Сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, В")
plt.title("FFT с фильтрацией частотной области")
plt.legend()
plt.grid()

# выводим график
plt.tight_layout()
plt.show()
