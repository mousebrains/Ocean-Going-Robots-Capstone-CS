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

def process(fn:str, args:argparse.ArgumentParser) -> None:
        meta = xr.open_dataset(fn, group="Meta") # For water depth
        wave = xr.open_dataset(fn, group="Wave")
        xyz = xr.open_dataset(fn, group="XYZ")

        depth = float(meta.WaterDepth)
        declination = float(meta.Declination)
        fs = float(xyz.SampleRate)
        print("Sampling Frequency", fs, "Hz Depth", depth, "m", "declination", declination)

        freq = wave.f.to_numpy()
        bandwidth = wave.Bandwidth[:].to_numpy()
        freqBounds = wave.FreqBounds[:,:].to_numpy()
        fMid = freqBounds.mean(axis=1) # Mid point of each band
        nFreq = freqBounds.shape[0]

        k = waveNumber(depth, fMid) # Wave number for each frequency at a depth

        for i in range(args.skip, min(args.skip + args.n, wave.t.size)):
            tw = wave.t[i]
            t0 = wave.TimeBounds[i,0]
            t1 = wave.TimeBounds[i,1]
            q = np.logical_and(xyz.t >= t0, xyz.t <= t1).to_numpy()
            if not q.any(): continue

            print("i", i, "q", q.sum(),
                    "t0", t0.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                    "t1", t1.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                    "tw", tw.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy())
            print("FlagPrimary", wave.FlagPrimary[i].to_numpy())
            print("FlagSecondary", wave.FlagSecondary[i].to_numpy())

            t = xyz.t[q].to_numpy() # Time of sample
            x = xyz.x[q].to_numpy() # x is northward
            y = xyz.y[q].to_numpy() # y is eastward
            z = xyz.z[q].to_numpy() # z is upwards

            # Denoising should happen here

            if args.detrend or args.demean:
                detrend = "constant" if args.demean else "linear"
                x = signal.detrend(x, type=detrend)
                y = signal.detrend(y, type=detrend)
                z = signal.detrend(z, type=detrend)

            # Get zero crossing points in z to get Tz, time between zero crossings
            Tz = zeroCrossingAverage(z, fs)

            xFFT = sp_fft.rfft(x, n=z.size) # northwards
            yFFT = sp_fft.rfft(y, n=z.size) # eastwards
            zFFT = sp_fft.rfft(z, n=z.size) # upwards
            f = sp_fft.rfftfreq(z.size, 1/fs) # Frequency of each FFT bin

            xxPSD = calcPSD(xFFT, xFFT, fs).real # Imaginary part is zero
            xyPSD = calcPSD(xFFT, yFFT, fs)
            # xzPSD = calcPSD(xFFT, zFFT, fs)

            # yxPSD = calcPSD(yFFT, xFFT, fs)
            yyPSD = calcPSD(yFFT, yFFT, fs).real # Imaginary part is zero
            # yzPSD = calcPSD(yFFT, zFFT, fs)

            zxPSD = calcPSD(zFFT, xFFT, fs)
            zyPSD = calcPSD(zFFT, yFFT, fs)
            zzPSD = calcPSD(zFFT, zFFT, fs).real # Imaginary part is zero

            q = np.logical_and(
                    np.less_equal.outer(freqBounds[:,0], f),
                    np.greater_equal.outer(freqBounds[:,1], f)
                    ) # Which frequency belongs to which band

            cnt = q.sum(axis=1) # Number of bins in each band

            xxBand = (q * xxPSD).sum(axis=1) / cnt # Mean of each band
            xyBand = (q * xyPSD).sum(axis=1) / cnt
            # xzBand = (q * xzPSD).sum(axis=1) / cnt # Not needed

            # yxBand = (q * yxPSD).sum(axis=1) / cnt
            yyBand = (q * yyPSD).sum(axis=1) / cnt
            # yzBand = (q * yzPSD).sum(axis=1) / cnt

            zxBand = (q * zxPSD).sum(axis=1) / cnt
            zyBand = (q * zyPSD).sum(axis=1) / cnt
            zzBand = (q * zzPSD).sum(axis=1) / cnt

            # Zeroth order
            a0 = zzBand

            # First order
            denom = np.sqrt(zzBand * (xxBand + yyBand))
            a1 =  zxBand.imag / denom
            b1 = -zyBand.imag / denom # minus from flipping east to west

            # Second order
            denom = xxBand + yyBand
            a2 = (xxBand - yyBand) / denom
            b2 = -2 * xyBand.real / denom # minus from flipping east to west

            # For deep water waves, the motion will be circular and K->1
            # otherwise K is a measure of the eccentricity.
            df = pd.DataFrame()
            df["q"] = np.linspace(0,100,11) # Quantiles at [0, 10, ..., 100] percent
            df["K"] = np.quantile(np.sqrt((xxBand + yyBand) / zzBand), df.q/100)
            print(df)

            # Spectral moments
            m0 = (a0 * bandwidth).sum()
            mm1 = (a0 * bandwidth / fMid).sum() # m_{-1}
            m1 = (a0 * bandwidth * fMid).sum()
            m2 = (a0 * bandwidth * fMid * fMid).sum()

            iDominant = a0.argmax() # Peak location

            # Centered Fourier coefficients
            theta0 = np.degrees(np.arctan2(b1, a1)) % 360 # Horizontal angle [0,360)
            theta0 = (theta0 + declination) % 360 # Magnetic to true
            Dp = theta0[iDominant]

            print("PeakPSD from CDIP", float(wave.PeakPSD[i]), "calc", a0.max())
            print("Hs from CDIP", float(wave.Hs[i]), 
                    "4*sqrt(z.var0)", 4 * np.sqrt(z.var()),
                    "4*sqrt(m0)", 4 * np.sqrt(m0))
            print("Tp from CDIP", float(wave.Tp[i]),
                    "calc", 1/fMid[iDominant],
                    "not banded", 1/f[zzPSD.argmax()])
            print("Ta from CDIP", float(wave.Ta[i]),
                    "from m0/m1", m0 / m1)
            print("Tz from CDIP", float(wave.Tz[i]),
                    "calc", Tz, "from m2(NOAA)", np.sqrt(m0/m2))
            print("T_E", mm1 / m0, "mean energy period")
            print("T_E/Tp", (mm1 / m0) / (1 / fMid[iDominant]), \
                    "wind(0.85-0.88) swell(0.93-0.97)")
            print("Dp", wave.Dp[i].to_numpy(), 
                    "from CDIP a1,b1",
                    float(np.degrees(np.arctan2(wave.B1[i,iDominant], wave.A1[i,iDominant]))) % 360,
                    "from calc a1,b1", Dp, "degrees")

            # Calculate centered Fourier Coefficients, m2 and n2 in geographic coordinates
            theta0 = np.radians(theta0) # Back to radians from degrees
            m1 = np.sqrt(a1**2 + b1**2)
            m2 =  a2 * np.cos(2 * theta0) + b2 * np.sin(2 * theta0)
            n2 =  a2 * np.sin(2 * theta0) + b2 * np.cos(2 * theta0)

            df = pd.DataFrame()
            df["f"] = fMid
            df["a1"] = a1
            df["b1"] = b1
            df["a2"] = a2
            df["b2"] = b2
            df["m1"] = m1
            df["m2"] = m2
            df["n2"] = n2
            df["spreadM1"] = np.sqrt(2 * (1 - df.m1))
            df["spreadM2"] = np.sqrt((1 - m2) / 2)
            df["kurtosis0"] = -df.n2 / df.spreadM2**3
            df["kurtosis1"] = (6 - 8 * m1 + 2 * m2) / (2 * (1 - m1))**2
            print(df)

            if args.plot:
                def plotit(ax, x, y, yTit, xTit=None, qX=False) -> None:
                    ax.plot(x, y, "-")
                    ax.grid()
                    ax.set_ylabel(yTit)
                    if xTit is not None: ax.set_xlabel(xTit)
                    if not qX: ax.get_xaxis().set_ticklabels([])

                (fig, ax) = plt.subplots(nrows=9, figsize=[9,9])
                plotit(ax[0], t, x, "North")
                plotit(ax[1], t, y, "East")
                plotit(ax[2], t, z, "Vert", "Date", True)

                plotit(ax[3], fMid, wave.A1[i, :], "A1")
                ax[3].plot(fMid, a1, "-")
                plotit(ax[4], fMid, wave.B1[i, :], "B1")
                ax[4].plot(fMid, b1, "-")
                plotit(ax[5], fMid, wave.A2[i, :], "A2")
                ax[5].plot(fMid, a2, "-")
                plotit(ax[6], fMid, wave.B2[i, :], "B1")
                ax[6].plot(fMid, b2, "-")
                plotit(ax[7], fMid, wave.M2[i, :], "m2")
                ax[7].plot(fMid, m2, "-")
                plotit(ax[8], fMid, wave.N2[i, :], "n2", "Frequency (Hz)", True)
                ax[8].plot(fMid, n2, "-")

                plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--skip", type=int, default=0, help="Number of chunks to skip")
parser.add_argument("--n", type=int, default=1, help="Number of chunks to process")
parser.add_argument("--plot", action="store_true", help="Show plots")
grp = parser.add_mutually_exclusive_group()
grp.add_argument("--detrend", action="store_true", help="Detrend data")
grp.add_argument("--demean", action="store_true", help="Demean data")
parser.add_argument("nc", nargs="+", type=str, help="netCDF file to process")
args = parser.parse_args()

for fn in args.nc:
    process(fn, args)
