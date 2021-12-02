#! /usr/bin/env python3
#
# Make a movie of a traveling sine wave in 2D
#
# Plot both the displacement and acceleration
#
# Nov-2021, Pat Welch, pat@mousebrains.com

import argparse
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def displacement(x:np.array, t:float, k:float, omega:float) -> np.array:
    return np.cos(k * x - omega * t)

def velocity(x:np.array, t:float, k:float, omega:float) -> np.array:
    return omega * np.sin(k * x - omega * t)

def acceleration(x:np.array, t:float, k:float, omega:float) -> np.array:
    return -omega*omega * np.cos(k * x - omega * t)

def initFunc() -> None:
    ax0.grid()
    ax0.set_xlabel("Distance k={} omega={}".format(args.k, args.omega))
    ax0.set_ylabel("Amplitude", color=color0)
    ax1.set_ylabel("Acceleration", color=color1)
    ax0.set_xlim(x.min(), x.max())
    ax0.set_ylim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax0.tick_params(axis="y", labelcolor=color0)
    ax1.tick_params(axis="y", labelcolor=color1)

def animate(t:float, *args) -> None:
    (x, line0, line1, tit, k, omega) = args
    tit.set_text("t={:5.2f}".format(t))
    line0.set_ydata(displacement(x, t, k, omega)) # Update the plot's data
    line1.set_ydata(acceleration(x, t, k, omega)) # Update the plot's data

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=float, default=1, help="Wave number")
parser.add_argument("--omega", type=float, default=1, help="Angular frequency")
parser.add_argument("--fps", type=int, default=30, help="Frames per second")
parser.add_argument("--duration", type=float, default=10, help="Time in seconds for the movie")
parser.add_argument("--bitrate", type=int, default=500, help="Output bits/sec")
parser.add_argument("--dpi", type=int, default=300, help="Output pixels/inch")
parser.add_argument("--save", action="store_true", help="Should the movie be saved?")
args = parser.parse_args()

(fn, _) = os.path.splitext(os.path.basename(sys.argv[0]))

color0 = "tab:blue"
color1 = "tab:green"

k = args.k
omega = args.omega

L = 2 * np.pi / k # Wave length

fig, ax0 = plt.subplots()
ax1 = ax0.twinx() # Second axes using the same x-axis

x = np.arange(0, 2 * L, L / 100)

line0, = ax0.plot(x, displacement(x, 0, k, omega), color=color0) # Initial line at t=0
line1, = ax1.plot(x, acceleration(x, 0, k, omega), color=color1) # Initial line at t=0

tit = ax0.set_title("t={:}".format(0)) # Initial title

dt = 1 / args.fps
times = np.arange(0, args.duration + dt/2, dt)

ani = animation.FuncAnimation(
    fig = fig, 
    func = animate, 
    frames = times,
    init_func = initFunc,
    fargs = (x, line0, line1, tit, k, omega), # Additional arguments fo func
    interval= np.ceil(1 / args.fps * 1000) if not args.save else 0,
    repeat = False,
    blit=False, 
    )

if args.save: # Save the animation
    ofn = fn + ".mp4"
    ani.save(ofn,
            fps=args.fps,
            bitrate=args.bitrate,
            dpi=args.dpi)
else: # Display
    print("Close plot to exit")
    plt.show()
