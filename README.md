# 4A) ASK
# Aim
Write a simple Python program for the modulation and demodulation of ASK 
# Tools required
Google Colab
# Program
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Parameters
fs = 1000
f_carrier = 50
bit_rate = 10

T = 1
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Message signal (binary data)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)

# Carrier signal
carrier = np.sin(2 * np.pi * f_carrier * t)

# ASK Modulation
ask_signal = message_signal * carrier

# ASK Demodulation
demodulated = ask_signal * carrier
filtered_signal = butter_lowpass_filter(demodulated, f_carrier, fs)
decoded_bits = (filtered_signal[::bit_duration] > 0.25).astype(int)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, carrier, label='Carrier Signal', color='g')
plt.title('Carrier Signal')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, ask_signal, label='ASK Modulated Signal', color='r')
plt.title('ASK Modulated Signal')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='k', marker='x')
plt.title('Decoded Bits')
plt.grid(True)

plt.tight_layout()
plt.show()

```
# Output Waveform

<img width="1190" height="790" alt="download" src="https://github.com/user-attachments/assets/d9c8736b-f4d6-4f70-a325-bfc0ca96a60d" />

# Results

<img width="1190" height="790" alt="download" src="https://github.com/user-attachments/assets/c7f34935-04c1-446f-b348-a5b2ff40c323" />

4B) FSK

# Aim
Write a simple Python program for the modulation and demodulation of FSK.

# Tools required
Google Colab

# Program
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Parameters
fs = 1000
f1 = 30
f2 = 70
bit_rate = 10
T = 1

t = np.linspace(0, T, int(fs * T), endpoint=False)

bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)

carrier_f1 = np.sin(2 * np.pi * f1 * t)
carrier_f2 = np.sin(2 * np.pi * f2 * t)

# FSK Modulation
fsk_signal = np.zeros_like(t)
for i, bit in enumerate(bits):
    start = i * bit_duration
    end = start + bit_duration
    freq = f2 if bit else f1
    fsk_signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

# Reference signals for demodulation
ref_f1 = np.sin(2 * np.pi * f1 * t)
ref_f2 = np.sin(2 * np.pi * f2 * t)

# Demodulation
corr_f1 = butter_lowpass_filter(fsk_signal * ref_f1, f2, fs)
corr_f2 = butter_lowpass_filter(fsk_signal * ref_f2, f2, fs)

decoded_bits = []
for i in range(bit_rate):
    start = i * bit_duration
    end = start + bit_duration
    energy_f1 = np.sum(corr_f1[start:end] ** 2)
    energy_f2 = np.sum(corr_f2[start:end] ** 2)
    decoded_bits.append(1 if energy_f2 > energy_f1 else 0)

decoded_bits = np.array(decoded_bits)
demodulated_signal = np.repeat(decoded_bits, bit_duration)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, fsk_signal, label='FSK Modulated Signal', color='r')
plt.title('FSK Modulated Signal')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, corr_f1, label='Filtered f1 Component', color='g')
plt.plot(t, corr_f2, label='Filtered f2 Component', color='m')
plt.title('Demodulated Components')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='k', marker='x')
plt.title('Decoded Bits')
plt.grid(True)

plt.tight_layout()
plt.show()
```
# Output Waveform

<img width="1190" height="790" alt="download (1)" src="https://github.com/user-attachments/assets/4ed51028-4b63-4304-8107-bac512144ea2" />


# Results

<img width="1190" height="790" alt="download (1)" src="https://github.com/user-attachments/assets/121a525a-2654-438f-ac6e-3d980d5e28b0" />



