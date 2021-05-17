import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import datetime
import calendar

stn = '179'
start_date = "05/21/2008 09:00"  # MM/DD/YYYY HH:MM
dataset = 'archive'  # Enter 'archive' or 'realtime'


# CDIP THREDDS OPeNDAP Dataset URL
# Archive
data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/' + \
    stn + 'p1/' + stn + 'p1_historic.nc'
# Realtime
if dataset == 'realtime':
    data_url = 'http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/realtime/' + stn + 'p1_rt.nc'

nc = netCDF4.Dataset(data_url)

sourceFile = open('demo.txt', 'w')
print(nc.variables, file=sourceFile)
sourceFile.close()
