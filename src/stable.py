#!/usr/bin/python

import tornado.web
import cv2

import calibration_worker
import stabilize
import numpy as np
import matplotlib.pyplot as plt

def render_trio(signal_x, signal_y, signal_z, timestamps):
    plt.plot(timestamps, signal_x, 'b-', timestamps, signal_y, 'g-', timestamps, signal_z, 'r-')
    plt.ylabel("Y")
    plt.show()

def main():
    gdf = calibration_worker.GyroscopeDataFile('/work/blueprints/c7/data/mockgyro.csv')
    gdf.parse()

    signal_x = gdf.get_signal_x()
    signal_y = gdf.get_signal_y()
    signal_z = gdf.get_signal_z()
    timestamps = gdf.get_timestamps()

    smooth_signal_x = stabilize.gaussian_filter(signal_x)
    smooth_signal_y = stabilize.gaussian_filter(signal_y)
    smooth_signal_z = stabilize.gaussian_filter(signal_z)

    # g is the difference between the smoothed version and the actual version
    g = [ [], [], [] ]
    g[0] = np.subtract(signal_x, smooth_signal_x).tolist()
    g[1] = np.subtract(signal_y, smooth_signal_y).tolist()
    g[2] = np.subtract(signal_z, smooth_signal_z).tolist()
    dgt = stabilize.diff(timestamps)

    theta = [ [], [], [] ]
    for component in [0, 1, 2]:
        sum_of_consecutives = np.add(g[component][:-1], g[component][1:])
        # The 2 is for the integration - and 10e9 for the nanosecond
        dx_0 = np.divide(sum_of_consecutives, 2 * 1000000000)
        num_0 = np.multiply(dx_0, dgt)
        theta[component] = [0]
        theta[component].extend(np.cumsum(num_0))

    render_trio(theta[0], theta[1], theta[2], timestamps)

if __name__ == "__main__":
    main()
