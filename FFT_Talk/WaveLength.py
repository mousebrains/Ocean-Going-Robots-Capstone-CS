#! /usr/bin/env python3
#
# For a given water depth and multiple periods, calculate the wavelength(s).
# Use a Newton-Raphson method to converge to a solution
#
# c = lambda / period = sqrt((g * lambda) / (2 pi) * tanh(2 pi depth / lambda)
#
# Where:
#  c is the wave phase speed
#  lambda is the wave length
#  period is the wave period
#  depth is the water depth
#  g is the gravitational acceleration
#    There is a latitude dependence of g which we'll ignore
#
# See http://web.mit.edu/13.021/demos/lectures/lecture19.pdf
#
# Phase speed
# For deep water waves
# lambda ~= g period^2 / 2pi
# For shallow water waves
# lambda ~= sqrt(g * depth) * period
#
#
# Use Newton-Raphson method to converge to a solution
# for:
#  L^2 / period^2 = (gL/(2pi)) * tanh(2pi depth / L)
#  L / period^2 = (g/(2pi)) * tanh(2pi depth / L)
#  g period^2 / (2 pi) = L / tanh(2pi depth / L)
#  2pi / (g * period^2) = tanh(2pi depth / L) / L
#  lhs = rhs
#  0 = lhs - rhs
#
# Oct-2020, Pat Welch, pat@mousebrains.com

import numpy as np

def waveLength(depth:float, period:np.array, **kwargs) -> np.array:
    # depth is in meters
    # period is in seconds
    # threshold is in meters
    # g gravitational acceleration m/sec^2
    # result is in meters

    items = {"threshold": 0.01, # m
            "maxIter": 10,  # count
            "g": 9.80665, # m/sec/sec
            }
    items.update(kwargs) # Overwrite from kwargs into items
    threshold = items["threshold"]
    maxIter = items["maxIter"]
    g = items["g"]

    twoPi = 2 * np.pi
    period2 = np.square(period) # Square of period
    twoPiDepth = twoPi * depth

    L = g * period2 / twoPi # Deep water approximation
    term = np.tanh(twoPi * depth / L) # see how close tanh is to one
    msk = abs(term) < 0.75 # Use shallow water approximation in these cases

    if msk.sum(): # Some shallow water apprixmations needed
        L[msk] = np.sqrt(g * depth) * period[msk] # Shallow water approximation

    lhs = twoPi / (g * period2)

    for cnt in range(maxIter): # Maximum number of iterations
        dL = (0.001 * L) / 2 # derivative step size
        xlhs = L - dL
        xrhs = L + dL
        y = lhs - np.tanh(twoPiDepth / L) / L
        ylhs = lhs - np.tanh(twoPiDepth / xlhs) / xlhs
        yrhs = lhs - np.tanh(twoPiDepth / xrhs) / xrhs
        dy = (ylhs - yrhs) / (xlhs - xrhs)
        dL = y / dy
        L -= dL
        if abs(dL).max() < threshold:
            break

    return L

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=float, metavar="meters", default=200,
            help="Water depth in meters")
    parser.add_argument("--minPeriod", type=float, metavar="seconds", default=1,
            help="Minimum Wave period in seconds")
    parser.add_argument("--maxPeriod", type=float, metavar="seconds", default=30,
            help="Maximum Wave period in seconds")
    parser.add_argument("--dPeriod", type=float, metavar="seconds", default=1,
            help="Seconds between min and max period in seconds")
    parser.add_argument("--threshold", type=float, metavar="meters",
            help="Convergence threshold")
    parser.add_argument("--iterations", type=int, metavar="int",
            help="Maximum number of iterations allowed for convergence")
    parser.add_argument("--g", type=float, metavar="g",
            help="Gravitational acceleration in m/sec^2")
    args = parser.parse_args()

    period = np.arange(args.minPeriod, args.maxPeriod + args.dPeriod, args.dPeriod)
    a = {}
    if args.threshold is not None: a["threshold"] = args.threshold
    if args.iterations is not None: a["maxIter"] = args.iterations
    if args.g is not None: a["g"] = args.g
    L = waveLength(args.depth, period, **a)

    print("Water depth ->", args.depth, "meters")
    a = pd.DataFrame()
    a["period"] = period
    a["lambda"] = L
    print(a)
