#! /usr/bin/env python3
#
# Plot the X, Y, and Z displacements for an entire file
#
# Feb-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import xarray as xr
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("nc", nargs=1, type=str, help="Input NetCDF file from CDIP")
args = parser.parse_args()

with xr.open_dataset(args.nc[0], group="XYZ") as ds:
    x = ds.x.copy()
    y = ds.y.copy()
    z = ds.z.copy()
    print(x.min(), y.min(), z.min())
    x[x < -999] = None
    y[y < -999] = None
    z[z < -999] = None
    (fig, axes) = plt.subplots(3)
    x.plot(ax=axes[0])
    y.plot(ax=axes[1])
    z.plot(ax=axes[2])
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    plt.suptitle("f{args.nc[0]} < -999 -> None")
    plt.show()
