#! /usr/bin/env python3
#
# Make a set of PNG plots of a Fourier transform for a displacement wave.
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--frequency", type=float, default=1/20, help="Frequency (Hz)")
parser.add_argument("--fs", type=float, default=10, help="Sampling frequency (Hz)")
parser.add_argument("--phase", type=float, default=-90, help="Phase (degrees)")
parser.add_argument("--amplitude", type=float, default=4, help="Sine wave amplitude")
parser.add_argument("--n", type=int, default=8192, help="Number of samples")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

omega = 2 * np.pi * args.frequency # Angular Frequency
tau = 1 / args.frequency # Period

dtSample = 1 / args.fs # Time step for sampling
dt = tau / 1000 # time step for plotting

duration = args.n * dtSample
tCurve  = np.linspace(0, duration, num=args.n * 100)
tSample = np.linspace(0, duration, num=args.n)

zCurve  = args.amplitude * np.cos(-omega * tCurve  + np.radians(args.phase))
zSample = args.amplitude * np.cos(-omega * tSample + np.radians(args.phase))

freq = np.fft.rfftfreq(tSample.shape[0], dtSample) # Frequency of each bin in rfft
fft = np.fft.rfft(zSample, norm="forward") # Only positive part of full FFT
fft[1:] *= 2 # Account for negative frequencies
fftAmp = np.abs(fft)

tit = f"f={args.frequency}" \
        + f" $\phi={args.phase}$" \
        + f" amp={args.amplitude}" \
        + f" $f_s={args.fs}$"

fig, ax = plt.subplots()
ax.plot(tSample, zSample, ".")
ax.plot(tCurve, zCurve, "-")
ax.grid()
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Displacement")
ax.set_title(tit)

if args.save:
    ofn = fn + ".spectrum.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

fig, ax = plt.subplots()
ax.plot(freq, fft.real, '-')
ax.plot(freq, fft.imag, '-')
ax.set_xlim(0, 2 * args.frequency)
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("a/b")
ax.set_title(tit)

if args.save:
    ofn = fn + ".fft.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

fig, ax = plt.subplots()
ax.plot(freq, fftAmp, '-')
ax.set_xlim(0, 2 * args.frequency)
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|fft|")
ax.set_title(tit)

if args.save:
    ofn = fn + ".abs.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)


fig, ax = plt.subplots()
ax.plot(freq, np.degrees(np.angle(fft)), '-')
ax.set_xlim(0, 2 * args.frequency)
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("$\phi$ (degrees)")
ax.set_title(tit)

if args.save:
    ofn = fn + ".phase.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)


if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()
