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
from WaveLength import waveLength

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=float, default=100, help="Water depth (m)")
parser.add_argument("--period", type=float, default=15, help="Wave period (s)")
parser.add_argument("--amplitude", type=float, default=2, help="Wave amplitude (m)")
parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

depth = args.depth
period = args.period
amplitude = args.amplitude

L = waveLength(depth, period)
k = 2 * np.pi / L

omega = 2 * np.pi / period

t = np.linspace(0, period, 1000)
z = amplitude * np.cos(-omega * t)
zA = -amplitude * omega * omega * np.cos(-omega * t)

x = amplitude / np.tanh(k * depth) * np.sin(-omega * t)
xA = -amplitude * omega * omega / np.tanh(k * depth) * np.sin(-omega * t)

fig, ax = plt.subplots()
tits = []
ax.plot(t, z, '-')
ax.plot(t, x, '-')
ax.plot(t, zA, '-')
ax.plot(t, xA, '-')
ax.grid()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.legend(("z", "x", "$a_z$", "$a_x$"))
ax.set_title(f"A={amplitude} period={period} h={depth} $\lambda={L:.1f}$")

if args.save:
    ofn = fn + ".png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()
