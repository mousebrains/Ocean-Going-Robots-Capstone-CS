#! /usr/bin/env python3
#
# Load the CDIP data that was fetched by extract.py
# and do some processing on it.
#
# URLs:
# https://docs.google.com/document/d/1Uz_xIAVD2M6WeqQQ_x7ycoM3iKENO38S4Bmn6SasHtY/pub
#
# Hs:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/compendium.html
#
# Hs Boxplot:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/annualHs_plot.html
#
# Sea Surface Temperature:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/temperature.html
#
# Polar Spectrum:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/polar.html
#
# Wave Direction and Energy Density by frequency bins:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/spectrum_plot.html
#
# XYZ displacements:
# https://cdip.ucsd.edu/themes/media/docs/documents/html_pages/dw_timeseries.html
#
# The Datawell documentation is very useful:
# https://www.datawell.nl/Portals/0/Documents/Manuals/datawell_manual_libdatawell.pdf
#
# Dec-2021, Pat Welch

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import scipy.fft as sp_fft
import sys
from WaveNumber import waveNumber

def calcPSD(xFFT:np.array, yFFT:np.array, fs:float) -> np.array:
    nfft = xFFT.size
    qOdd = nfft % 2
    n = (nfft -  qOdd) * 2 # Number of data points input to FFT
    psd = (xFFT.conjugate() * yFFT) / (fs * n)
    if not qOdd:       # Even number of FFT bins
        psd[1:] *= 2   # Real FFT -> double for non-zero freq
    else:              # last point unpaired in Nyquist freq
        psd[1:-1] *= 2 # Real FFT -> double for non-zero freq
    return psd

def zeroCrossingAverage(z:np.array, fs:float) -> float:
    q = z[0:-1] * z[1:] < 0 # opposite times between successive zs
    iLHS = np.flatnonzero(q) # Indices of LHS
    iRHS = iLHS + 1
    zLHS = z[iLHS]
    zRHS = z[iRHS]
    zFrac = -zLHS / (zRHS - zLHS) # Fraction of zRHS,zLHS interval to zero from zLHS
    tZero = zFrac / fs # Seconds from iLHS to the zero crossing point
    dt = np.diff(iLHS) / fs # interval between iLHS
    dt += zFrac[1:] / fs # Add in time from RHS to zero crossing
    dt -= zFrac[0:-1] / fs # Take off tiem from LHS to zero crossing
    return 2 * dt.mean() # 2 times the half wave zero crossing average time

def calcAcceleration(x:np.array, fs:float) -> np.array:
    x = x.copy() # Local copy
    x[x<-999.9] = None
    dx2 = np.zeros(x.shape)
    dx2[2:] = np.diff(np.diff(x))
    dx2[0:2] = dx2[2]
    return dx2 * fs * fs

def process(fn:str, args:argparse.ArgumentParser) -> None:
        meta = xr.open_dataset(fn, group="Meta") # For water depth
        wave = xr.open_dataset(fn, group="Wave")
        xyz = xr.open_dataset(fn, group="XYZ")

        depth = float(meta.WaterDepth)
        declination = float(meta.Declination)
        fs = float(xyz.SampleRate)
        print("Sampling Frequency", fs, "Hz Depth", depth, "m", "declination", declination)

        if args.meta and len(args.meta) > 0: # Output meta data
            df = pd.DataFrame()
            df["fs"] = [fs]
            df["latitude"] = [float(meta.DeployLatitude)]
            df["longitude"] = [float(meta.DeployLongitude)]
            df["depth"] = [depth]
            df["declination"] = [declination]
            df.to_csv(args.meta, index=False)

        if args.acceleration and len(args.acceleration) > 0: # Output acceleration data
            df = pd.DataFrame()
            df["t"] = xyz.t
            df["ax"] = calcAcceleration(xyz.x.to_numpy(), fs)
            df["ay"] = calcAcceleration(xyz.y.to_numpy(), fs)
            df["az"] = calcAcceleration(xyz.z.to_numpy(), fs)
            df.to_csv(args.acceleration, index=False)

        if args.displacement and len(args.displacement) > 0: # Output displacement data
            df = pd.DataFrame()
            df["t"] = xyz.t
            df["x"] = xyz.x
            df["y"] = xyz.y
            df["z"] = xyz.z
            df.to_csv(args.displacement, index=False)

        if args.wave and len(args.wave) > 0: # Output anlaysis 
            df = pd.DataFrame()
            df["tLower"] = wave.TimeBounds[:,0]
            df["tUpper"] = wave.TimeBounds[:,1]
            df["Hs"] = wave.Hs
            df["Ta"] = wave.Ta
            df["Tp"] = wave.Tp
            df["Tz"] = wave.Tz
            df["Dp"] = wave.Dp
            df["PeakPSD"] = wave.PeakPSD
            df.to_csv(args.wave, index=False)

        if args.freq and len(args.freq) > 0: # Output frequency bins
            df = pd.DataFrame()
            df["Bandwidth"] = wave.Bandwidth
            df["fLower"] = wave.FreqBounds[:,0]
            df["fUpper"] = wave.FreqBounds[:,1]
            df.to_csv(args.freq, index=False)


parser = argparse.ArgumentParser()
parser.add_argument("--meta", type=str, default="meta.csv", help="Metadata CSV filename")
parser.add_argument("--acceleration", type=str, default="acceleration.csv",
        help="Acceleration CSV filename")
parser.add_argument("--displacement", type=str, default="displacement.csv",
        help="Displacement CSV filename")
parser.add_argument("--wave", type=str, default="wave.csv", help="Wave CSV filename")
parser.add_argument("--freq", type=str, default="freq.csv", help="Frequency CSV filename")
parser.add_argument("nc", nargs=1, type=str, help="netCDF file to process")
args = parser.parse_args()

for fn in args.nc:
    process(fn, args)
