# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:42:24 2023

@author: ASM
"""

import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
c = 3e8
BW = 1e6
fs = 2 * BW
T = 1e-3
t = np.arange(0, T, (1/fs))

# Range and Doppler calculation
rang = 3000
vel = 5
dop = 2 * vel / (c / 10e9)
n_pulses = 1000
SNR = 10

# Transmitted signal
sig_tx = np.cos(np.pi * (BW / T) * t * t) + 1j * np.sin(np.pi * (BW / T) * t * t)
sig_tx = sig_tx / np.mean(abs(sig_tx)**2)

# Initialize array for received signals
Sig_rx = np.empty((n_pulses, sig_tx.size)) + 1j * np.empty((n_pulses, sig_tx.size))

# Loop through each pulse
for p in range(n_pulses):
    r = rang + (p - 1) * vel * 2 * T
    delay_num = round(2 * fs * r / c)
    
    # Create received signal for the current pulse
    sig_rd = np.concatenate((np.zeros(int(delay_num)) + 1j * np.zeros(int(delay_num)),
                             sig_tx * (np.cos(2 * np.pi * dop * t) + 1j * np.sin(2 * np.pi * dop * t))))
    
    # Apply doppler-dependent phase shift and add noise
    phase_shift = np.cos(2 * np.pi * dop * p * 2 * T) + 1j * np.sin(2 * np.pi * dop * p * 2 * T)
    noise = np.random.normal(0, 10 ** (-SNR / 10) / np.sqrt(2), sig_tx.size) + \
            1j * np.random.normal(0, 10 ** (-SNR / 10) / np.sqrt(2), sig_tx.size)
    Sig_rx[p, :] = sig_rd[:sig_tx.size] * np.repeat(phase_shift, sig_tx.size) + noise

# Visualize the received signal for a specific pulse (e.g., pulse index 0)
selected_pulse_index = 500
plt.figure(figsize=(10, 6))
plt.plot(np.real(Sig_rx[selected_pulse_index]), label='Real')
plt.plot(np.imag(Sig_rx[selected_pulse_index]), label='Imaginary')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title(f'Received Signal for Pulse {selected_pulse_index}')
plt.legend()
plt.grid()
plt.show()
