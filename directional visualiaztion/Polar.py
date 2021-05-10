# Polar.py
#
# Using Ocean Going Robots to Observe Wave Conditions
# Directional Visualization Framework
# 
# Benjamin Lee, Mayinger
# mayingeb@oregonstate.edu
#
#
# How to call this script on the command line:
# python Polar.py <your data>.csv



# imports
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm



def main():

	
	# INPUT VALIDATION ##########################################################

	# too many arguments are given 
	if len(sys.argv) > 2:
		print("Error: too many arguments are given.")
		print("Script must be called with the following format: python Polar.py <your data>.csv")
		return

	# too few arguments are given
	if len(sys.argv) < 2:
		print("Error: too few arguments are given.")
		print("Script must be called with the following format: python Polar.py <your data>.csv")
		return

	# check if input is CSV file
	if sys.argv[1].find(".csv") == 0:
		print("Error: CSV file required.")
		return

	# file_name is set to second argument, see script usage at the top
	data_file = sys.argv[1]


	# check if file exists, if not print error message
	try:
		open(data_file, "r")

	except IOError:
		print("Error: the provided input file doesn't exist.")
		return

	min_period = 0.025	# define max wave period (small Hz value)
	max_period = 0.56	# define min wave period (large Hz value)

	# define data array, 56 x 72 matrix
	data = np.loadtxt(data_file, delimiter=',')

	# make call to polar_plot to visualize the input data
	polar_plot(data, min_period, max_period)

	return

###################################################################################################
# polar_plot() provides a graphical representation of polar data using MatPlotLib
# arguments: data = 56 x 72 float matrix, max_period = small Hz value, min_period = large Hz value
# output: produces a new window pop-out window displaying a polar plot
def polar_plot(data, min_period, max_period):

	# define coordinates

	# thetas
	azimuths = np.radians(np.linspace(0, 360, data.shape[1]))	# theta values

	# radii
	zeniths = np.linspace(0.025, 0.58, data.shape[0])	# radius values

	# create variables theta and r as 2D matrices from meshing zeniths and azimuths
	theta, r = np.meshgrid(azimuths, zeniths)

	# set up our plot as a polar projection
	fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

	# set plot title
	fig.suptitle('Polar Wave Spectrum', fontsize=18, fontweight='bold')

	# set plot directional labels
	# ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

	ax.set_theta_zero_location("N") # North at top
	ax.set_theta_direction("clockwise") # East/90 to the right


	# set radius grid values
	rTicks = np.array([20, 12, 8, 6, 4])

	ax.set_rgrids(radii=1/rTicks, 
			labels=map(lambda x: "{} s".format(x), rTicks),
			angle=0, # along the North axis
			color="darkorange",
			fontweight="bold",
			fontsize=11) # Color of the labels

	# set theta grid values
	ax.set_thetagrids(np.arange(30, 360+1, 30), labels=None)

	ax.tick_params(color="red")
	
	ax.set_facecolor("black")

	# threshold declaration for colormap low cut-off
	threshold = np.amax(data) * 0.02
	# threshold = 2e-3

	# number of levels within our contour
	nLevels = 1000

	# create contour levels manually using defined threshold
	zMax = data.max()
	levelsAbove = np.linspace(threshold, zMax, nLevels) # levels above threshold
	dAbove = np.mean(np.diff(levelsAbove)) # Mean difference between levels
	levelsBelow = np.arange(0, threshold, dAbove)
	levels = np.append(levelsBelow, levelsAbove)

	# create colormap for contour
	jet = cm.get_cmap("jet", nLevels) # Blue to red above threshold
	azure = clr.ListedColormap(["azure"]) # Below threshold color
	cmap = np.row_stack((
		azure(np.arange(levelsBelow.size)), 
		jet(np.arange(nLevels)))) # Color map with azure below threshold

	ax.tick_params(axis="x", direction="in", pad=-19) # Theta labels inside plot


	# use contourf to create the heat map
	colorax = ax.contourf(theta, r, data, levels=levels, colors=cmap)

	# polar plot limits, creates empty inner circle
	# ax.set_rlim(top=1/3.2, bottom=-0.08)
	#ax.set_rlim(top=1/3.2, bottom=-0.08)
	ax.set_rlim(top=1/3, bottom=-0.08)

	# color bar creation
	cbar = fig.colorbar(colorax, orientation="horizontal", ticks=(0, zMax/2, zMax))
	cbar.set_label('Energy Density (m*m/Hz/deg)', fontsize=16, loc="center")
	cbar.ax.get_yaxis().labelpad = 30

	# show plot
	plt.show()

	return


# call main
main()