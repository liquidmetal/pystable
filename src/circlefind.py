#!/usr/bin/python

# This file writes out a file called circles.pickle
# Use this file as input for intrinsic.py

import cv2
import sys, pickle

cornerlist = []
patternsize = (4, 11)
cnt = 0
for filepath in sys.argv[1:]:
    print("Processing: %s" % filepath)

    img = cv2.imread(filepath)
    shape = img.shape

    smallsize = (shape[1]/8, shape[0]/8)
    small = cv2.resize(img, smallsize)
    ret, corners = cv2.findCirclesGrid(small, patternsize, None, cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret:
        cornerlist.append(corners.tolist())
        cv2.drawChessboardCorners(small, patternsize, corners, ret)
        cv2.imwrite("./test%02d.png" % cnt, small)
    else:
        print("    Circles not found")

    cnt+=1

with open("pickle.circles", "w") as fp:
    pickle.dump(cornerlist, fp)
