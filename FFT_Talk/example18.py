#! /usr/bin/env python3 #
#
# Horizontal wave motion
#
# http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from WaveLength import waveLength

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=float, default=100, help="Water depth (m)")
parser.add_argument("--period", type=float, default=15, help="Wave period (s)")
parser.add_argument("--amplitude", type=float, default=2, help="Wave amplitude (m)")
parser.add_argument("--theta", type=float, default=15, help="Wave heading (deg)")
parser.add_argument("--fs", type=float, default=10, help="Sampling frequency (Hz)")
parser.add_argument("--n", type=int, default=8192, help="Number of sampling bins")
parser.add_argument("--noise", type=float, default=0.5, help="Gaussian noise amplitude")
parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian noise width")
parser.add_argument("--seed", type=int, default=123456, help="Random seed")
parser.add_argument("--average", type=str, default="mean",
        choices=["mean", "median"], help="Peridogram averaging")
parser.add_argument("--scaling", type=str, default="spectrum",
        choices=["density", "spectrum"], help="Peridogram scaling")
parser.add_argument("--window", type=str, default="boxcar",
        choices=["boxcar", "triang", "blackman", "hamming", "hann", "bartlett", "flattop",
            "parzen", "bohman", "blackmanharris", "nuttal", "barthann", "cosine", "exponetial",
            "tukey", "taylor"],
        help="Peridogram windowing")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

rng = np.random.default_rng(args.seed)

depth = args.depth
period = args.period
amplitude = args.amplitude
theta = args.theta
fs = args.fs
n = args.n

L = waveLength(depth, period)
k = 2 * np.pi / L

omega = 2 * np.pi / period

t = np.linspace(0, n / fs, n)

zA     = -amplitude * omega * omega * np.cos(-omega * t)
xAccel = -amplitude * omega * omega / np.tanh(k * depth) * np.sin(-omega * t)

sTheta = np.sin(np.radians(theta))
cTheta = np.cos(np.radians(theta))
xA =  cTheta * xAccel # Rotate from along wave space  into East/West space
yA = -sTheta * xAccel # Rotate from along wave space into North/South space

# Add in noise
xA += args.noise * rng.normal(scale=args.sigma, size=xA.shape)
yA += args.noise * rng.normal(scale=args.sigma, size=yA.shape)
zA += args.noise * rng.normal(scale=args.sigma, size=zA.shape)

# Sxy => S23
# Szx => S12
# Szy => S13

(freq, Sxy) = signal.csd(xA, yA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )
(_, Szx) = signal.csd(zA, xA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )
(_, Szy) = signal.csd(zA, yA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )
(_, Sxx) = signal.csd(xA, xA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )
(_, Syy) = signal.csd(yA, yA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )
(_, Szz) = signal.csd(zA, zA,
        fs=fs, 
        nperseg=n,
        nfft=n,
        window=args.window,
        scaling=args.scaling,
        average=args.average,
        )

a0 =  np.real(Szz) / ((2 * np.pi * freq)**4 * np.pi)
a1 = -np.imag(Szx) / ((2 * np.pi * freq)**2 * np.pi * k)
b1 = -np.imag(Szy) / ((2 * np.pi * freq)**2 * np.pi * k)
a2 = (np.real(Sxx) - np.real(Syy)) / (k**2 * np.pi)
b2 = 2 * np.real(Sxy) / (k**2 * np.pi)

r1 = np.sqrt(a1**2 + b1**2) / a0
r2 = np.sqrt(a2**2 + b2**2) / a0
theta1 = np.arctan2(b1, a1)
theta2 = np.arctan2(b2, a2)

th = np.outer(np.radians(np.arange(0, 360)), np.ones(theta1.shape))
D = (0.5 \
        + r1 * np.cos(    (th - theta1)) \
        + r2 * np.cos(2 * (th - theta2)) \
        ) / np.pi
print(D.shape)
print(th.shape)
print(theta1.shape)
print(r1.shape)
fig, ax = plt.subplots()
tits = []
ax.plot(D, '-')
ax.grid()
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("a0 (m)")
ax.set_title(f"A={amplitude} period={period} h={depth} $\lambda={L:.1f}$, fs={fs}, n={n}\n" + \
        f"noise={args.noise} sigma={args.sigma} seed={args.seed}")
ax.set_xlim(0.5/period, 2/period)
# ax.set_ylim(-0.5, 0.5)

if args.save:
    ofn = fn + ".png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()
