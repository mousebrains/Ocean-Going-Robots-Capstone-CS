# Ocean-Going-Robots-Capstone-CS
Repository for OSU CS capstone group 40

This system takes in raw wave data and will display the directional (polar plot) and nondirectional (histogram). 

## How to Run, Execute and Understand the System
 - Clone the repository

      `cd {repo name}`
 - You are now in the newly created directory

### Include the required python modules
 - In the command line, run 
 
     `pip install pipreqs`
     
     `pipreqs .`
     
     `pip install -r requirements.txt`
 
 - You will now have all required python modules installed

### Getting the Code up and Running
  - To run the code, run the following command in the command line
  
      `python3 Driver.py --csv={data file}`
      
  - Where {data file} is a .csv of a time series of displacements in the x,y,z direction for a given station
  - If python3 is not downloaded, the latest release can be found here https://www.python.org/downloads/
  - You will then see two windows pop up. The polar plot is the directional data, and the histogram is the nondirectional

## Overview of Files

### Driver.py
 - The main interface of the system that is able to call all other submodules

### Logger.py
- Documents averages, maximums, minimums, and outliers from inputted data for easier debugging

### NonDirectionalVisualization.py
- Create a histogram plot of the nondirectional wave data. Wave energy density (m^2/Hz) on the y-axis and Wave Frequency (Hz) x-axis

### Polar.py
- Creates a polar plot of the directional data. The plot is set up so that the degrees shown are direction (N, S, E, W) i.e. 90° is due east. Periods (seconds) and Energy density (m^2/Hz/deg)

### spectralAnalysis.py
- Estimates the directional spectrum from wave readings

### waveParameters.py
- Calculates directional/nondirectional wave parameters from spectral data

