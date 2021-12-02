#! /usr/bin/env python3 #
#
# Calculate significant wave height from periodogram for multiple waves
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def acceleration(t:np.array, 
        amplitude:np.array, phase:np.array, omega:np.array, 
        rng, noise:float, sigma:float) -> np.array:
    aCos = np.cos(-np.outer(omega, t) + np.outer(np.radians(phase), np.ones(t.shape)))
    norm = -np.outer(amplitude * omega * omega, np.ones(t.shape))
    eta = noise * rng.normal(scale=sigma, size=t.shape)
    return eta + (norm * aCos).sum(axis=0)

parser = argparse.ArgumentParser()
parser.add_argument("--period", type=float, action="append", help="Wave period(s) in s")
parser.add_argument("--amplitude", type=float, action="append", help="Sine wave amplitude(s)")
parser.add_argument("--phase", type=float, action="append", help="Phase(s) (degrees)")
parser.add_argument("--fs", type=float, default=10, help="Sampling frequency (Hz)")
parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise amplitude")
parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian noise width")
parser.add_argument("--seed", type=int, default=123456, help="Random seed")
parser.add_argument("--scaling", type=str, default="spectrum",
        choices=["density", "spectrum"], help="Peridogram scaling")
parser.add_argument("--n", type=int, default=8192, help="FFT window width")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

if args.period is None:
    n = len(args.amplitude) if args.amplitude is not None else \
            len(args.phase) if args.phase is not None else 2
    args.period = 10 * (np.arange(0, n) + 1)
if args.amplitude is None: args.amplitude = np.arange(0, len(args.period)) + 1
if args.phase is None: args.phase = np.linspace(-90, 90, len(args.period))

if len(args.period) != len(args.amplitude):
    parser.error("You must specify the same number of --period and --amplitude options")
if len(args.period) != len(args.phase):
    parser.error("You must specify the same number of --period and --phase options")

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

rng = np.random.default_rng(args.seed)

omega = 2 * np.pi / np.array(args.period)
amplitude = np.array(args.amplitude)
phase = np.array(args.phase)

dtSample = 1 / args.fs # Time step for sampling

duration = args.n * dtSample

tCurve  = np.linspace(0, duration, num=args.n * 100)
tSample = np.linspace(0, duration, num=args.n)

zCurve   = acceleration(tCurve,  amplitude, phase, omega, rng, 0, 0)
zSample  = acceleration(tSample, amplitude, phase, omega, rng, args.noise, args.sigma)

(freq, PSD) = signal.periodogram(zSample, fs=args.fs, scaling=args.scaling)

tit = f"$period={args.period}$" \
        + f" amp={args.amplitude}" \
        + f" $\phi={args.phase}$" \
        + f" $f_s={args.fs}$" \
        + "\n" \
        + f" $\eta_A={args.noise}$" \
        + f" $\eta_\sigma={args.sigma}$" \
        + f" scaling={args.scaling}" \
        + f" seed={args.seed}" \
        + f" n={args.n}" \

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
ax.plot(freq, PSD, "o-")
for i in range(len(args.period)):
    tau = args.period[i]
    f = 1 / tau
    j = np.argmin(np.abs(freq - (1 / tau)))
    jj = j # np.arange(j-1, j+2)
    C11 = PSD[jj].sum()
    Hm0 = np.sqrt(C11) / np.square(2 * np.pi * freq[j]) # Normlize due to acceleration
    Hm00 = amplitude[i] / np.sqrt(2) # Expected value (RMS of amplitude)
    ax.plot(freq[j], PSD[j], "*", color="tab:orange")
    ax.text(freq[j], PSD[j],
            "  RMS\n" +
            f"  Expected = {Hm00:.2f}\n" +
            f"  Measured = {Hm0:.2f}",
            verticalalignment="top",
            )
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density $((m/s^2)^2)$")
ax.set_title(tit)
ax.set_xlim(0, 2 / min(args.period))

if args.save:
    ofn = fn + ".psd.png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()
