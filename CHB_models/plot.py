#!/usr/bin/env python3
"""
plot_hbridge_bin.py
Reads hbridge_output.bin and plots:
1. Time-domain waveforms
2. FFT magnitude spectra for each cell.
"""

import numpy as np
import matplotlib.pyplot as plt
import struct

def main():
    fname = "hbridge_output.bin"

    with open(fname, "rb") as f:
        # Read header: two ints and a float
        header = f.read(4 + 4 + 4)
        Ncells, Nsamp, fs = struct.unpack("iif", header)

        # Read waveform data
        data = np.fromfile(f, dtype=np.float32, count=Ncells * Nsamp)

    # Reshape to (Ncells, Nsamp)
    waveforms = data.reshape(Ncells, Nsamp)
    t = np.arange(Nsamp) / fs

    # ----- Time-domain plot -----
    plt.figure(figsize=(10, 5))
    for c in range(Ncells):
        plt.plot(t, waveforms[c], label=f"Cell {c}")
    plt.title("H-Bridge Cell Output Voltages")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # ----- FFT magnitude plots -----
    # Frequency vector (one-sided)
    freqs = np.fft.rfftfreq(Nsamp, 1/fs)

    plt.figure(figsize=(10, 5))
    for c in range(Ncells):
        # Compute one-sided FFT magnitude (normalize by Nsamp)
        fft_vals = np.fft.rfft(waveforms[c])
        mag = np.abs(fft_vals) / Nsamp
        plt.plot(freqs, mag, label=f"Cell {c}")
    plt.title("FFT Magnitude of H-Bridge Outputs")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [V]")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
