#!/usr/bin/env python

import scipy
import numpy as np
import math
import cv2
from scipy.interpolate import griddata

def get_gaussian_kernel(sigma2, v1, v2, normalize=True):
    gauss = [math.exp(-(float(x*x) / sigma2)) for x in range(v1, v2+1)]
    total = sum(gauss)

    if normalize:
        gauss = [x/total for x in gauss]

    return gauss
    

def gaussian_filter(input_array):
    """
    """
    # Step 1: Define the convolution kernel
    kernel = get_gaussian_kernel(4000, -120, 120)

    # Step 2: Convolve
    return np.convolve(input_array, kernel, 'same')

def diff(timestamps):
    """
    Returns differences between consecutive elements
    """
    return np.ediff1d(timestamps)

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out

    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """
    Generic method to return a rotation matrix
    - +ve angle = anticlockwise
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])

    R  = np.diag([cosa, cosa, cosa])
    R +=  np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([ [0.0,           -direction[2],  direction[1]],
                    [direction[2],  0.0,           -direction[0]],
                    [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)

    return M

def rotx_matrix(angle):
    return rotation_matrix(angle, [1, 0, 0])

def roty_matrix(angle):
    return rotation_matrix(angle, [0, 1, 0])

def rotz_matrix(angle):
    return rotation_matrix(angle, [0, 0, 1])


if __name__ == '__main__':
    rx = rotx_matrix(math.radians(45))
    ry = roty_matrix(math.radians(45))
    rz = rotz_matrix(math.radians(45))

    print rz * rx * ry

