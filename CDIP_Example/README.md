# Work with CDIP data

- `extract.py` Fetched from CDIP and save into a NetCDF file
- `process.py` Process one of the fetched NetCDF files 

To extract data use a command like:

`./extract.py --stn=067 --nhours=4 --start="2020-12-25 12:00"`
which will save data to `067.20201225_1200.20201225_1600.nc`

You can use `ncdump -h 067.20201225_1200.20201225_1600.nc` to view the data structure within the NetCDF file.

To process the above file:

`./process.py --plot --demean 067.20201225_1200.20201225_1600.nc`
which takes out the mean of the x/y/z displacements, prints some diagnostic information and plots.

<pre>
Sampling Frequency 1.2799999713897705 Hz Depth 262.0 m declination 11.930000305175781
i 0 q 2048 t0 2020-12-25 12:00:00 t1 2020-12-25 12:26:40 tw 2020-12-25 12:00:00
FlagPrimary 1
FlagSecondary 0
        q         K
0     0.0  0.622644
1    10.0  0.850814
2    20.0  0.907958
3    30.0  0.954644
4    40.0  0.981394
5    50.0  1.031048
6    60.0  1.074512
7    70.0  1.108055
8    80.0  1.199648
9    90.0  1.446701
10  100.0  4.239456
PeakPSD from CDIP 5.653039455413818 calc 6.742757797241211
Hs from CDIP 1.2999999523162842 4*sqrt(z.var0) 1.336582064628601 4*sqrt(m0) 1.3448458664054819
Tp from CDIP 20.0 calc 19.99999970197678 not banded 19.2771088646119
Ta from CDIP 10.752382278442383 from m1(CDIP) 10.810459255711951
Tz from CDIP 8.510638236999512 calc 9.031467135383542 from m2(NOAA) 8.5956397596759
Dp 201.84375 from CDIP a1,b1 201.84375 from calc a1,b1 201.09061351073368
</pre>

* No attempt has been made to take out the noise

- `WaveNumber.py` calculates the wave number given a period and water depth
- `general.py` calculates wave parameters for displacement, velocity, or acceleration
- `velocity.py` calculates wave parameters for velocity
- `acceleration.py` calculates wave parameters for acceleration
- `acc2csv.py` extracts NetCDF data into CSV files
  - `displacement.csv` is the CDIP displacement data
  - `acceleration.csv` is the acceleration as calculated from the CDIP displacement data
  - `meta.csv` is the CDIP metadata
  - `wave.csv` is the CDIP wave data
  - `freq.csv` is the CDIP wave frequency bin data
- `header.dump` is the output of `ncdump -h 067.20220102_0000.20220102_0600.nc`
