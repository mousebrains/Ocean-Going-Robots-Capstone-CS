#! /usr/bin/env python3 #
#
# Ocean wave length versus depth
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from WaveLength import waveLength

parser = argparse.ArgumentParser()
parser.add_argument("--minDepth", type=float, default=10, help="Starting water depth (m)")
parser.add_argument("--maxDepth", type=float, default=250, help="Starting water depth (m)")
parser.add_argument("--nDepth", type=int, default=100, help="Number of depth bins")

parser.add_argument("--minPeriod", type=float, default=5, help="Starting period (s)")
parser.add_argument("--maxPeriod", type=float, default=30, help="Starting period (s)")
parser.add_argument("--nPeriod", type=int, default=5, help="Number of period bins")

parser.add_argument("--dpi", type=int, default=300, help="Figure pixels/inch")
parser.add_argument("--save", action="store_true", help="Should output PNG be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

periods = np.linspace(args.minPeriod, args.maxPeriod, args.nPeriod)
depth = np.linspace(args.minDepth, args.maxDepth, args.nDepth)

L = []
for i in range(len(depth)):
    L.append(waveLength(depth[i], periods))

L = np.array(L)

fig, ax = plt.subplots()
tits = []
for i in range(len(periods)):
    ax.plot(L[:,i], depth, '-')
    tits.append(f"{periods[i]}")
ax.grid()
ax.set_xlabel("Wave Length (m)")
ax.set_ylabel("Depth (m)")
ax.legend(tits)

if args.save:
    ofn = fn + ".png"
    print("Saving to", ofn)
    plt.savefig(ofn, dpi=args.dpi)

if not args.save: # Plot to screen
    print("Close plot to exit")
    plt.show()
