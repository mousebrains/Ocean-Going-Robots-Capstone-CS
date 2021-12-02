#! /usr/bin/env python3
#
# Make a set of PNG plots of a Fourier transform of a noisy acceleration.
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
parser.add_argument("--noise", type=float, default=0.5, help="Gaussian noise amplitude")
parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian noise width")
parser.add_argument("--seed", type=int, default=123456, help="Random seed")
parser.add_argument("--n", type=int, default=8192, help="Number of samples")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

rng = np.random.default_rng(args.seed)

omega = 2 * np.pi * args.frequency # Angular Frequency
tau = 1 / args.frequency # Period

dtSample = 1 / args.fs # Time step for sampling
dt = tau / 1000 # time step for plotting

duration = args.n * dtSample
tCurve  = np.linspace(0, duration, num=args.n * 100)
tSample = np.linspace(0, duration, num=args.n)

zCurve  = -args.amplitude * omega*omega * np.cos(-omega * tCurve  + np.radians(args.phase))
zSample = -args.amplitude * omega*omega * np.cos(-omega * tSample + np.radians(args.phase))
zSample += args.noise * rng.normal(scale=args.sigma, size=zSample.shape)

freq = np.fft.rfftfreq(tSample.shape[0], dtSample) # Frequency of each bin in rfft
fftOmega = 2 * np.pi * freq # Angular frequency of each rfft bin
fft = np.fft.rfft(zSample, norm="forward") # Only positive part of full FFT
fft[1:] *= 2 # Account for negative frequencies
fft[1:] /= fftOmega[1:] * fftOmega[1:] # Drop out omega^2 term to get original A term, ignore DC term
fftAmp = np.abs(fft)

tit = f"f={args.frequency}" \
        + f" $\phi={args.phase}$" \
        + f" amp={args.amplitude}" \
        + f" $f_s={args.fs}$" \
        + f" $\eta_A={args.noise}$" \
        + f" $\eta_\sigma={args.sigma}$" \
        + f" seed={args.seed}"

fig, ax = plt.subplots()
ax.plot(tSample, zSample, ".")
ax.plot(tCurve, zCurve, "-")
ax.grid()
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Acceleration")
ax.set_title(tit)

if args.save:
    ofn = fn + ".spectrum.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

fig, ax = plt.subplots()
ax.plot(freq[1:], fft[1:].real, '-')
ax.plot(freq[1:], fft[1:].imag, '-')
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
ax.plot(freq[1:], fftAmp[1:], '-')
ax.set_xlim(0, 2 * args.frequency)
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("$|fft| / \omega^2_n$")
ax.set_title(tit)

if args.save:
    ofn = fn + ".abs.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)


fig, ax = plt.subplots()
ax.plot(freq[1:], np.degrees(np.angle(fft[1:])), '-')
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
