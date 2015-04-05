#!/usr/bin/python

import cv2, sys, pickle
import numpy as np

def get_physical_positions(width, height, squareSize=1.0):
    corners = []
    for i in xrange(0, height*2):
        for j in xrange(0, width, 2):
            x = j + i%2
            if x >= width:
                continue

            y = i

            corners.append( (float(x*squareSize), float(y*squareSize), 0) )

    return corners
            

def main():
    circlelist = []
    with open(sys.argv[1], 'r') as fp:
        circlelist = pickle.load(fp)

    physical_corners = get_physical_positions(11, 4)
    
    num_images = len(circlelist)
    image_points = np.array(circlelist, dtype=np.float32, ndmin=2)
    object_points = np.array([physical_corners for x in range(num_images)], dtype=np.float32, ndmin=2)

    import pdb; pdb.set_trace()

    image_size = (520, 390)
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(np.array(object_points), np.array(image_points), image_size)

    output_dict = {'ret': ret,
                    'camera_matrix': cameraMatrix,
                    'dist_coeffs': distCoeffs,
                    'rvecs': rvecs,
                    'tvecs': tvecs}

    with open('intrincs.pickle', 'w') as fp:
        pickle.dump(output_dict, fp)

if __name__ == '__main__':
    main()
