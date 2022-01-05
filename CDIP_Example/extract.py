#! /usr/bin/env python3
#
# Get CDIP NetCDF data and store it as a Pandas dataframe
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
# Dec-2021, Pat Welch

import argparse
import datetime
import os
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import datetime
import sys

class CDIP:
    def __init__(self, args:argparse.ArgumentParser) -> None:
        self.args = args
        self.url = self.mkURL()
        print(self.url)

        self.stime = np.datetime64(args.start)
        self.etime = self.stime + np.timedelta64(args.nhours, "h")

        self.ofn = os.path.join(args.outdir, 
                args.stn
                + "." + self.stime.astype(datetime.datetime).strftime("%Y%m%d_%H%M")
                + "." + self.etime.astype(datetime.datetime).strftime("%Y%m%d_%H%M")
                + ".nc")

        self.nc = netCDF4.Dataset(self.url)

    @staticmethod
    def addArgs(parser:argparse.ArgumentParser) -> None:
        parser.add_argument("--outdir", type=str, default=".", help="Output directory")
        parser.add_argument("--stn", type=str, required=True, help="CDIP station number")
        grp = parser.add_mutually_exclusive_group()
        grp.add_argument("--archive", action="store_true", help="Fetch archive files")
        grp.add_argument("--realtime", action="store_true", help="Fetch current/realtime files")
        parser.add_argument("--deployment", type=int,
                help="Which archived deployment to retrieve data from")
        parser.add_argument("--nhours", type=int, default=2,
                help="Number of hourse of data to fetch")
        parser.add_argument("--start", type=str,
                default=(datetime.date.today()-datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                help="Ending date to retrieve data from, yyyy-mm-dd HH:MM:SS")
        parser.add_argument("--url", type=str,
                default="https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip",
                help="Base URL to fetch data from")

    def timeMask(self, key:str) -> tuple[np.array, np.array]:
        t = self.nc[key][:].astype("datetime64[s]") # UTC seconds
        q = np.logical_and(t >= self.stime, t <= self.etime)
        if q.any():
            return (t, q)
        print(f"No data found for {key} in the interval {self.stime} to {self.etime}")
        return (None, None)

class RT(CDIP):
    def __init__(self, args:argparse.ArgumentParser) -> None:
        CDIP.__init__(self, args)
        self.minTime = None
        self.maxTime = None
        self.attributes()
        self.meta()
        self.Wave()
        self.SST()
        self.GPS()
        self.DWR()

    def mkURL(self) -> str:
        stn = self.args.stn
        url = self.args.url
        if args.archive:
            return url + "/archive/" + stn + "p1/" + stn + "p1_historic.nc"
        # realtime
        return url + "/realtime/" + stn + "p1_rt.nc"

    def attributes(self) -> None:
        ds = xr.Dataset(attrs=self.nc.__dict__)
        ds.to_netcdf(self.ofn, mode="w") # Save attributes

    def meta(self) -> None:
        grp = "Meta"
        nc = self.nc
        ds = xr.Dataset({
            "name": nc["metaStationName"][:].tobytes().strip(b"\x00").strip().decode("utf-8"),
            "DeployLatitude": nc["metaDeployLatitude"][:],
            "DeployLongitude": nc["metaDeployLongitude"][:],
            "WaterDepth": nc["metaWaterDepth"][:],
            "Declination": nc["metaDeclination"][:],
            })
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)

    def Wave(self) -> None:
        grp = "Wave"
        nc = self.nc
        (t, q) = self.timeMask("waveTime")
        if t is None: return
        timeBounds = nc["waveTimeBounds"][q][:].astype("datetime64[s]")
        self.minTime = timeBounds.min()
        self.maxTime = timeBounds.max()
        ds = xr.Dataset({
            "FlagPrimary": ("t", nc["waveFlagPrimary"][q]),
            "FlagSecondary": ("t", nc["waveFlagSecondary"][q]),
            "Hs": ("t", nc["waveHs"][q]),
            "Ta": ("t", nc["waveTa"][q]),
            "Tp": ("t", nc["waveTp"][q]),
            "Tz": ("t", nc["waveTz"][q]),
            "Dp": ("t", nc["waveDp"][q]),
            "PeakPSD": ("t", nc["wavePeakPSD"][q]),
            "SourceIndex": ("t", nc["waveSourceIndex"][q]),
            "FreqFlagPrimary": ("f", nc["waveFrequencyFlagPrimary"][:]),
            "FreqFlagSecondary": ("f", nc["waveFrequencyFlagSecondary"][:]),
            "Bandwidth": ("f", nc["waveBandwidth"][:]),
            "EnergyDensity": (("t", "f"), nc["waveEnergyDensity"][q][:]),
            "MeanDirection": (("t", "f"), nc["waveMeanDirection"][q][:]),
            "A1": (("t", "f"), nc["waveA1Value"][q][:]),
            "B1": (("t", "f"), nc["waveB1Value"][q][:]),
            "A2": (("t", "f"), nc["waveA2Value"][q][:]),
            "B2": (("t", "f"), nc["waveB2Value"][q][:]),
            "M2": (("t", "f"), nc["waveM2Value"][q][:]),
            "N2": (("t", "f"), nc["waveM2Value"][q][:]),
            "CheckFactor": (("t", "f"), nc["waveCheckFactor"][q][:]),
            "Spread": (("t", "f"), nc["waveSpread"][q][:]),
            "TimeBounds": (("t", "bounds"), timeBounds),
            "FreqBounds": (("f", "bounds"), nc["waveFrequencyBounds"][:][:]),
            },
            coords={
                "f": ("f", nc["waveFrequency"][:]),
                "t": ("t", t[q]),
                "bounds": ("bounds", np.arange(0, 2)),
                },
            )
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)

    def SST(self) -> None:
        grp = "SST"
        nc = self.nc
        (t, q) = self.timeMask("sstTime")
        if t is None: return
        ds = xr.Dataset({
            "FlagPrimary": ("t", nc["sstFlagPrimary"][q]),
            "FlagPrimary": ("t", nc["sstFlagPrimary"][q]),
            "T": ("t", nc["sstSeaSurfaceTemperature"][q]),
            "RefT": ("t", nc["sstReferenceTemp"][q]),
            "TimeBounds": (("t", "bounds"), nc["sstTimeBounds"][q][:].astype("datetime64[s]")),
            },
            coords={
                "t": ("t", t[q]),
                "bounds": ("bounds", np.arange(0, 2)),
                },
            )
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)

    def GPS(self) -> None:
        grp = "GPS"
        nc = self.nc
        (t, q) = self.timeMask("gpsTime")
        if t is None: return
        ds = xr.Dataset({
            "Flag": ("t", nc["gpsStatusFlags"][q]),
            "Latitude": ("t", nc["gpsLatitude"][q]),
            "Longitude": ("t", nc["gpsLongitude"][q]),
            "SourceIndex": ("t", nc["gpsSourceIndex"][q]),
            "TimeBounds": (("t", "bounds"), nc["gpsTimeBounds"][q][:].astype("datetime64[s]")),
            },
            coords={
                "t": ("t", t[q]),
                "bounds": ("bounds", np.arange(0, 2)),
                },
            )
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)

    def DWR(self) -> None:
        grp = "DWR"
        nc = self.nc
        (t, q) = self.timeMask("dwrTime")
        if t is None: return
        ds = xr.Dataset({
            "BatteryLevel": ("t", nc["dwrBatteryLevel"][q]),
            "BatteryWeeksOfLife": ("t", nc["dwrBatteryWeeksOfLife"][q]),
            "zAccelerometerOffset": ("t", nc["dwrZAccelerometerOffset"][q]),
            "xAccelerometerOffset": ("t", nc["dwrXAccelerometerOffset"][q]),
            "yAccelerometerOffset": ("t", nc["dwrYAccelerometerOffset"][q]),
            "Orientation": ("t", nc["dwrOrientation"][q]),
            "Inclination": ("t", nc["dwrInclination"][q]),
            "SourceIndex": ("t", nc["dwrSourceIndex"][q]),
            "TimeBounds": (("t", "bounds"), nc["dwrTimeBounds"][q][:].astype("datetime64[s]")),
            },
            coords={
                "t": ("t", t[q]),
                "bounds": ("bounds", np.arange(0, 2)),
                },
            )
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)

class XYZ(CDIP):
    def __init__(self, args:argparse.ArgumentParser,
            minTime:np.datetime64, maxTime:np.datetime64) -> None:
        CDIP.__init__(self, args)
        grp = "XYZ"
        nc = self.nc
        nc.set_auto_mask(False)
        t0 = nc["xyzStartTime"][0].astype("datetime64[s]")
        fs = nc["xyzSampleRate"][0]
        delay = nc["xyzFilterDelay"][0]
        x = nc["xyzXDisplacement"]
        t = t0 + (np.arange(0, x.size) / fs * 1e9).astype("timedelta64[ns]")
        q = np.logical_and(t >= minTime, t <= maxTime)
        if not q.any():
            print(f"No data found for {grp} in the interval {self.stime} to {self.etime}")
            return
        print("Extracting", q.sum(), "points for", grp)
        ds = xr.Dataset({
            "StartTime": t0,
            "SampleRate": fs,
            "FilterDelay": nc["xyzFilterDelay"],
            "x": ("t", nc["xyzXDisplacement"][q]),
            "y": ("t", nc["xyzYDisplacement"][q]),
            "z": ("t", nc["xyzZDisplacement"][q]),
            "FlagPrimary": ("t", nc["xyzFlagPrimary"][q]),
            "FlagSecondary": ("t", nc["xyzFlagSecondary"][q]),
            "SourceIndex": ("t", nc["xyzSourceIndex"][q]),
            },
            coords={"t": ("t", t[q])},
            )
        print("Saving", grp, "to", self.ofn)
        ds.to_netcdf(self.ofn, mode="a", group=grp)
    
    def mkURL(self) -> str:
        stn = self.args.stn
        url = self.args.url
        if args.archive:
            return url + "/archive/" + stn + "p1/" + stn + "p1_d" \
                    + str(self.args.deployment) + ".nc"
        # realtime
        return url + "/realtime/" + stn + "p1_xy.nc"

parser = argparse.ArgumentParser()
CDIP.addArgs(parser)
args = parser.parse_args()

if args.archive and args.deployment is None:
    parser.error("If you specify --archive you must also supply --deployment")

os.makedirs(args.outdir, mode=0o755, exist_ok=True) # Create directory recursively

rt = RT(args) # Must be done first to create fresh netCDF file and get time bounds for xyz
xyz = XYZ(args, rt.minTime, rt.maxTime)
