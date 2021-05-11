import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def nonDirectional(df):
    Fq = df.loc[:,"freq Hz"]
    Dmean = df.loc[:,"Dmean deg"]
    Ed = df.loc[:,"energy m*m/Hz"]

    # Create figure and specify figure size
    plt.figure(figsize=(15, 15))

    # Create 2 stacked subplots for Energy Density (Ed) and Mean Direction (Dmean)
    pEd = plt.subplot(2, 1, 1)
    pEd.step(Fq[:], Ed[:], marker='o', where='mid')
    pEd.fill_between(Fq[:], Ed[:], alpha=0.5, step="mid")
    pDmean = plt.subplot(2, 1, 2, sharex=pEd)
    pDmean.plot(Fq[:], Dmean[:],
                color='crimson', marker='o', linestyle="")

    # Set title
    # Add more info about buoy
    plt.suptitle("Oceanside, CA\n", fontsize=22, y=0.95)

    # Set tick parameters
    pEd.tick_params(axis='y', which='major', labelsize=12, right='off')
    pDmean.tick_params(axis='y', which='major', labelsize=12, right='off')

    # Make secondary x- and y-axes for each graph. Shows both Frequency and Period for x-axes.
    pEd2y = pEd.twiny()  # Copy x-axis for Graph #1
    pDmean2y = pDmean.twiny()  # Copy x-axis for Graph #2

    # Set axis limits for each plot
    ymax = np.ceil(max(Ed[:]))
    pEd.set_xlim(0, 0.6)
    pEd.set_ylim(0, ymax)
    pEd2y.set_xlim(0, 0.6)
    pDmean.set_ylim(0, 360)
    pDmean.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    pDmean2y.set_xlim(0, 0.6)

    # Label each axis
    pEd.set_xlabel('Frequency (Hz)', fontsize=14, x=0.3)
    pEd.set_ylabel('Energy density (m^2/Hz)', fontsize=18)
    pDmean.set_ylabel('Direction (deg)', fontsize=18)
    pDmean2y.set_xlabel('Period (s)', fontsize=14, x=0.7)

    ## Format top axis labels to show 'Period' values at tickmarks corresponding to 'Frequency' x-axs
    # Top subplot
    pEd2y.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    pEd2y.set_xticklabels(['10', '5', '3.3', '2.5', '2.0', '1.7'])

    # Bottom subplot
    pDmean2y.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    pDmean2y.set_xticklabels(['10', '5', '3.3', '2.5', '2.0', '1.7'])

    # Plot dashed gridlines
    pEd.grid(b=True, which='major', axis='x', alpha=0.3, linestyle='-')
    pEd.grid(b=True, which='major', axis='y', alpha=0.3, linestyle='-')
    pDmean.grid(b=True, which='major', axis='x', alpha=0.3, linestyle='-')
    pDmean.grid(b=True, which='major', axis='y', alpha=0.3, linestyle='-')

    plt.show()
