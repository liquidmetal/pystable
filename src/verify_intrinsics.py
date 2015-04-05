#!/usr/bin/python

import cv2, sys, pickle

intrinsics = {}
with open('pickle.intrinsics', 'r') as fp:
    intrinsics = pickle.load(fp)

import pdb; pdb.set_trace()
