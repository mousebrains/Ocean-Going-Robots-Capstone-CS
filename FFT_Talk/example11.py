#! /usr/bin/env python3
#
# Show different FFT widths on a noise-free cosine wave
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--frequency", type=float, default=1/20, help="Frequency (Hz)")
parser.add_argument("--fs", type=float, default=10, help="Sampling frequency (Hz)")
parser.add_argument("--phase", type=float, default=-90, help="Phase (degrees)")
parser.add_argument("--amplitude", type=float, default=4, help="Sine wave amplitude")
parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise amplitude")
parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise width")
parser.add_argument("--seed", type=int, default=123456, help="Random seed")
parser.add_argument("--scaling", type=str, default="spectrum",
        choices=["density", "spectrum"], help="Peridogram scaling")
parser.add_argument("--n", type=int, action="append", help="Number of samples")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

if args.n is None:
    args.n = [512, 1024, 2048, 4096, 8192]

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

rng = np.random.default_rng(args.seed)

omega = 2 * np.pi * args.frequency # Angular Frequency
tau = 1 / args.frequency # Period

dtSample = 1 / args.fs # Time step for sampling

duration = max(args.n) * dtSample

tCurve  = np.linspace(0, duration, num=max(args.n) * 100)
tSample = np.linspace(0, duration, num=max(args.n))

zCurve  = -args.amplitude * omega*omega * np.cos(-omega * tCurve  + np.radians(args.phase))
zSample = -args.amplitude * omega*omega * np.cos(-omega * tSample + np.radians(args.phase))
zSample += args.noise * rng.normal(scale=args.sigma, size=zSample.shape)

tit = f"f={args.frequency}" \
        + f" $\phi={args.phase}$" \
        + f" amp={args.amplitude}" \
        + f" $f_s={args.fs}$" \
        + f" $\eta_A={args.noise}$" \
        + f" $\eta_\sigma={args.sigma}$" \
        + "\n" \
        + f" scaling={args.scaling}" \
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
cmap = ["tab:blue", "tab:orange", "tab:green", "tab:red", 
        "tab:purple", "tab:brown", "tab:pink", "tab:gray",
        "tab:olive", "tab:cyan"]

tits = []
for i in range(len(args.n)): # Walk through the different FFT window widths
    n = args.n[i]
    tits.append(str(n))
    (freq, PSD) = signal.periodogram(zSample[0:n], fs=args.fs, scaling=args.scaling)
    zz = np.sqrt(PSD) * np.sqrt(2) / np.square(2 * np.pi * freq)

    ax.plot(freq, PSD, "o-", color=cmap[i])

ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density $((m/s^2)^2)$")
ax.legend(tits, loc="best")
ax.set_title(tit)
ax.set_xlim(0, 2 * args.frequency)
ax.set_ylim(0, 0.16)

if args.save:
    ofn = fn + ".psd.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()